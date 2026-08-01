using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using System.Runtime.CompilerServices;
using StardewModdingAPI;
using StardewModdingAPI.Events;
using StardewValley;
using StardewValley.Minigames;

namespace JunimoKartRLBridge;

public sealed class ModEntry : Mod
{
    private const int ProgressMode = 2;
    private const int InfiniteMode = 3;
    private readonly object sync = new();
    private Config config = new();
    private BridgeServer? server;
    private BridgeSnapshot latestSnapshot = IdleSnapshot("Bridge has not ticked yet.", 0);
    private bool desiredJumpHeld;
    private bool actualJumpHeld;
    private string? pendingStartMode;
    private bool pendingAdvance;
    private int autoAdvanceTitleTicksRemaining;
    private int snapshotVersion;

    public override void Entry(IModHelper helper)
    {
        this.config = helper.ReadConfig<Config>();
        helper.WriteConfig(this.config);

        helper.Events.GameLoop.UpdateTicked += this.OnUpdateTicked;
        helper.Events.GameLoop.ReturnedToTitle += this.OnReturnedToTitle;
        helper.ConsoleCommands.Add(
            "jkrl_start",
            "Start Junimo Kart through the RL bridge.\n\nUsage: jkrl_start [progress|endless]",
            (_, args) => this.RequestStart(args.FirstOrDefault() ?? "progress")
        );
        helper.ConsoleCommands.Add(
            "jkrl_release",
            "Release the bridge-held jump input.",
            (_, _) =>
            {
                lock (this.sync)
                {
                    this.desiredJumpHeld = false;
                }
            }
        );

        if (this.config.StartServerOnLaunch)
        {
            this.server = new BridgeServer(this, this.Monitor, this.config);
            this.server.Start();
        }
    }

    internal BridgeResponse HandleBridgeRequest(ClientRequest? request)
    {
        if (request is null)
            return BridgeResponse.Error("Empty or invalid request.", this.GetLatestSnapshot());

        var type = (request.Type ?? "state").Trim().ToLowerInvariant();
        switch (type)
        {
            case "ping":
                return BridgeResponse.Success("pong", this.GetLatestSnapshot(), "Junimo Kart RL Bridge is alive.");

            case "state":
                return BridgeResponse.Success("state", this.GetLatestSnapshot());

            case "start":
            case "reset":
                this.RequestStart(request.Mode ?? "progress");
                return BridgeResponse.Success("accepted", this.GetLatestSnapshot(), $"Start requested: {request.Mode ?? "progress"}.");

            case "action":
                if (request.Jump.HasValue)
                {
                    lock (this.sync)
                    {
                        this.desiredJumpHeld = request.Jump.Value;
                    }
                }
                return BridgeResponse.Success("state", this.GetLatestSnapshot());

            case "continue":
            case "advance":
                lock (this.sync)
                {
                    this.pendingAdvance = true;
                }
                return BridgeResponse.Success("state", this.GetLatestSnapshot());

            default:
                return BridgeResponse.Error($"Unknown request type '{request.Type}'.", this.GetLatestSnapshot());
        }
    }

    internal BridgeSnapshot GetLatestSnapshot()
    {
        lock (this.sync)
        {
            return this.latestSnapshot;
        }
    }

    private void RequestStart(string mode)
    {
        lock (this.sync)
        {
            this.pendingStartMode = mode;
            this.desiredJumpHeld = false;
            this.autoAdvanceTitleTicksRemaining = this.config.AutoAdvanceTitleAfterStart
                ? Math.Max(this.config.AutoAdvanceTitleTicks, 0)
                : 0;
        }
    }

    private void OnReturnedToTitle(object? sender, ReturnedToTitleEventArgs e)
    {
        lock (this.sync)
        {
            this.desiredJumpHeld = false;
            this.actualJumpHeld = false;
            this.pendingStartMode = null;
            this.pendingAdvance = false;
            this.autoAdvanceTitleTicksRemaining = 0;
            this.latestSnapshot = IdleSnapshot("Returned to title.", ++this.snapshotVersion);
        }
    }

    private void OnUpdateTicked(object? sender, UpdateTickedEventArgs e)
    {
        this.EnsureRunsWhenUnfocused();

        if (!Context.IsWorldReady)
        {
            lock (this.sync)
            {
                this.latestSnapshot = IdleSnapshot("World is not ready. Load a save first.", ++this.snapshotVersion);
            }
            return;
        }

        string? startMode;
        bool advance;
        bool desiredJump;
        lock (this.sync)
        {
            startMode = this.pendingStartMode;
            this.pendingStartMode = null;
            advance = this.pendingAdvance;
            this.pendingAdvance = false;
            desiredJump = this.desiredJumpHeld;
        }

        if (startMode is not null)
            this.StartMineCart(startMode);

        if (Game1.currentMinigame is MineCart mineCart)
        {
            if (advance)
                this.ForceProgressModeGameplay(mineCart, newGameFromTitle: true);
            this.AutoContinueNonGameplayStates(mineCart);
            this.ApplyJump(mineCart, desiredJump);
            var snapshot = this.CreateSnapshot(mineCart, desiredJump);
            lock (this.sync)
            {
                this.latestSnapshot = snapshot;
            }
            return;
        }

        if (this.actualJumpHeld)
            this.actualJumpHeld = false;

        lock (this.sync)
        {
            this.latestSnapshot = IdleSnapshot("Current minigame is not Junimo Kart.", ++this.snapshotVersion);
        }
    }

    private void EnsureRunsWhenUnfocused()
    {
        if (!this.config.ForceRunWhenUnfocused)
            return;

        if (Game1.options is not null && Game1.options.pauseWhenOutOfFocus)
            Game1.options.pauseWhenOutOfFocus = false;
    }

    private void StartMineCart(string mode)
    {
        var normalized = mode.Trim().ToLowerInvariant();
        var modeId = normalized is "endless" or "infinite" ? InfiniteMode : ProgressMode;
        this.actualJumpHeld = false;
        this.desiredJumpHeld = false;
        this.autoAdvanceTitleTicksRemaining = this.config.AutoAdvanceTitleAfterStart
            ? Math.Max(this.config.AutoAdvanceTitleTicks, 0)
            : 0;
        Game1.currentMinigame = new MineCart(0, modeId);
        this.Monitor.Log($"Started Junimo Kart mode '{normalized}' (mode id {modeId}).", LogLevel.Info);
    }

    private void AutoContinueNonGameplayStates(MineCart mineCart)
    {
        if (!this.config.AutoContinueProgressModeNonGameplayStates && this.autoAdvanceTitleTicksRemaining <= 0)
            return;

        var gameMode = ReflectionUtil.Field<int>(mineCart, "gameMode");
        if (gameMode != ProgressMode)
        {
            this.autoAdvanceTitleTicksRemaining = 0;
            return;
        }

        this.ForceProgressModeGameplay(mineCart, newGameFromTitle: true);
    }

    private void ForceProgressModeGameplay(MineCart mineCart, bool newGameFromTitle)
    {
        var gameState = ReflectionUtil.EnumId(ReflectionUtil.Field(mineCart, "gameState"));
        switch (gameState)
        {
            case 0: // Title
                if (!this.config.AutoContinueProgressModeNonGameplayStates)
                {
                    if (this.autoAdvanceTitleTicksRemaining <= 0)
                        return;
                    this.autoAdvanceTitleTicksRemaining--;
                }

                if (newGameFromTitle)
                    ReflectionUtil.Invoke(mineCart, "restartLevel", true);
                else
                    ReflectionUtil.Invoke(mineCart, "ShowCutscene");

                this.autoAdvanceTitleTicksRemaining = 0;
                break;

            case 3: // Map
                ReflectionUtil.Invoke(mineCart, "ShowCutscene");
                break;

            case 4: // Cutscene
                ReflectionUtil.Invoke(mineCart, "EndCutscene");
                break;
        }
    }

    private void ApplyJump(MineCart mineCart, bool desiredJump)
    {
        var player = ReflectionUtil.Field(mineCart, "player");
        ReflectionUtil.SetField(mineCart, "isJumpPressed", desiredJump);
        bool grounded = ReflectionUtil.BoolMethod(player, "IsGrounded", ReflectionUtil.Field<bool>(player, "_grounded"));

        if (desiredJump)
        {
            // A rising edge while airborne used to QueueJump and could silently
            // buffer a jump for the next landing. Real player control requires a
            // fresh press from the ground, so airborne re-presses are ignored.
            if (!this.actualJumpHeld && grounded)
                ReflectionUtil.Invoke(player, "QueueJump");
        }
        else
        {
            if (this.actualJumpHeld)
                ReflectionUtil.Invoke(player, "ReleaseJump");
        }

        this.actualJumpHeld = desiredJump;
    }

    private BridgeSnapshot CreateSnapshot(MineCart mineCart, bool desiredJump)
    {
        var player = ReflectionUtil.Field(mineCart, "player");
        var playerPosition = ReflectionUtil.VectorField(player, "position");
        var playerVelocity = ReflectionUtil.VectorField(player, "velocity");
        var currentTrackType = ReflectionUtil.Field(player, "currentTrackType");
        var gameMode = ReflectionUtil.Field<int>(mineCart, "gameMode");
        var levelsBeat = ReflectionUtil.Field<int>(mineCart, "levelsBeat");
        var gameOver = ReflectionUtil.Field<bool>(mineCart, "gameOver");
        var reachedFinish = ReflectionUtil.Field<bool>(mineCart, "reachedFinish");
        var grounded = ReflectionUtil.BoolMethod(player, "IsGrounded", ReflectionUtil.Field<bool>(player, "_grounded"));

        var snapshot = new BridgeSnapshot
        {
            InMinigame = true,
            MinigameType = mineCart.GetType().FullName ?? "MineCart",
            Version = ++this.snapshotVersion,
            Score = ReflectionUtil.Field<int>(mineCart, "score"),
            LevelsBeat = levelsBeat,
            GameMode = gameMode,
            LivesLeft = ReflectionUtil.Field<int>(mineCart, "livesLeft"),
            CurrentTheme = ReflectionUtil.Field<int>(mineCart, "currentTheme"),
            GameState = ReflectionUtil.EnumId(ReflectionUtil.Field(mineCart, "gameState")),
            ReachedFinish = reachedFinish,
            GameOver = gameOver,
            Completed = gameMode == ProgressMode && levelsBeat >= 6,
            GamePaused = ReflectionUtil.Field<bool>(mineCart, "gamePaused"),
            JumpHeld = desiredJump,
            TotalTime = ReflectionUtil.DoubleProperty(mineCart, "totalTime"),
            TotalTimeMs = ReflectionUtil.DoubleProperty(mineCart, "totalTimeMS"),
            SecondsOnThisLevel = ReflectionUtil.Field<float>(mineCart, "secondsOnThisLevel"),
            ScreenLeftBound = ReflectionUtil.Field<float>(mineCart, "screenLeftBound"),
            CheckpointPosition = ReflectionUtil.Field<float>(mineCart, "checkpointPosition"),
            DistanceToTravel = ReflectionUtil.Field<int>(mineCart, "distanceToTravel"),
            Player = new PlayerSnapshot
            {
                Position = ReflectionUtil.Vector(playerPosition),
                Velocity = ReflectionUtil.Vector(playerVelocity),
                Bounds = ReflectionUtil.Bounds(player),
                Grounded = grounded,
                Jumping = ReflectionUtil.BoolMethod(player, "IsJumping", ReflectionUtil.Field<bool>(player, "_jumping")),
                JumpReady = grounded && !this.actualJumpHeld,
                CurrentTrackType = currentTrackType?.ToString() ?? "",
                CurrentTrackTypeId = ReflectionUtil.EnumId(currentTrackType)
            }
        };

        snapshot.TracksAhead = this.GetTracksAhead(mineCart, playerPosition.X);
        snapshot.EntitiesAhead = this.GetEntitiesAhead(mineCart, playerPosition.X, player);
        return snapshot;
    }

    private List<TrackSnapshot> GetTracksAhead(MineCart mineCart, float playerX)
    {
        var tracks = ReflectionUtil.Enumerate(ReflectionUtil.Field(mineCart, "_tracks"));
        return tracks
            .Select(track => this.TrackSnapshot(track, playerX))
            .Where(track => track.Dx >= -this.config.LookbehindPixels && track.Dx <= this.config.LookaheadPixels)
            .OrderBy(track => track.Dx)
            .ThenBy(track => track.Y)
            .Take(Math.Max(this.config.MaxTracks, 0))
            .ToList();
    }

    private TrackSnapshot TrackSnapshot(object track, float playerX)
    {
        var position = ReflectionUtil.VectorField(track, "position");
        var trackType = ReflectionUtil.Field(track, "trackType");
        var obstacle = ReflectionUtil.Field(track, "obstacle");
        return new TrackSnapshot
        {
            X = position.X,
            Y = position.Y,
            Dx = position.X - playerX,
            Bounds = ReflectionUtil.Bounds(track),
            Type = trackType?.ToString() ?? "",
            TypeId = ReflectionUtil.EnumId(trackType),
            HasObstacle = obstacle is not null,
            ObstacleType = obstacle?.GetType().Name,
            ObstacleBounds = ReflectionUtil.Bounds(obstacle)
        };
    }

    private List<EntitySnapshot> GetEntitiesAhead(MineCart mineCart, float playerX, object? player)
    {
        var entities = ReflectionUtil.Enumerate(ReflectionUtil.Field(mineCart, "_entities"));
        return entities
            .Where(entity => !ReferenceEquals(entity, player) && !ReflectionUtil.InheritsTypeName(entity, "Track"))
            .Select(entity => this.EntitySnapshot(entity, playerX))
            .Where(entity => entity.Dx >= -this.config.LookbehindPixels && entity.Dx <= this.config.LookaheadPixels)
            .OrderBy(entity => entity.Dx)
            .ThenBy(entity => entity.Y)
            .Take(Math.Max(this.config.MaxEntities, 0))
            .ToList();
    }

    private EntitySnapshot EntitySnapshot(object entity, float playerX)
    {
        var position = ReflectionUtil.VectorField(entity, "position");
        return new EntitySnapshot
        {
            Id = RuntimeHelpers.GetHashCode(entity),
            Type = entity.GetType().Name,
            X = position.X,
            Y = position.Y,
            Dx = position.X - playerX,
            Visible = ReflectionUtil.Field<bool>(entity, "visible", true),
            Enabled = ReflectionUtil.Field<bool>(entity, "enabled", true),
            IsObstacle = ReflectionUtil.InheritsTypeName(entity, "Obstacle"),
            IsPickup = ReflectionUtil.InheritsTypeName(entity, "Pickup"),
            Bounds = ReflectionUtil.Bounds(entity)
        };
    }

    private static BridgeSnapshot IdleSnapshot(string message, int version)
    {
        return new BridgeSnapshot
        {
            InMinigame = false,
            MinigameType = "",
            Message = message,
            Version = version
        };
    }
}
