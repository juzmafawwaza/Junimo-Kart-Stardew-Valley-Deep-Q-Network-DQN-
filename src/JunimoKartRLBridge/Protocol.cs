using System.Text.Json;

namespace JunimoKartRLBridge;

internal static class Protocol
{
    public static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false
    };
}

internal sealed class ClientRequest
{
    public string? Type { get; set; }
    public string? Mode { get; set; }
    public bool? Jump { get; set; }
}

internal sealed class BridgeResponse
{
    public bool Ok { get; set; }
    public string Type { get; set; } = "state";
    public string? Message { get; set; }
    public BridgeSnapshot? State { get; set; }

    public static BridgeResponse Success(string type, BridgeSnapshot? state, string? message = null)
    {
        return new BridgeResponse
        {
            Ok = true,
            Type = type,
            Message = message,
            State = state
        };
    }

    public static BridgeResponse Error(string message, BridgeSnapshot? state = null)
    {
        return new BridgeResponse
        {
            Ok = false,
            Type = "error",
            Message = message,
            State = state
        };
    }
}

internal sealed class BridgeSnapshot
{
    public bool InMinigame { get; set; }
    public string MinigameType { get; set; } = "";
    public string? Message { get; set; }
    public int Version { get; set; }
    public int Score { get; set; }
    public int LevelsBeat { get; set; }
    public int GameMode { get; set; }
    public int LivesLeft { get; set; }
    public int CurrentTheme { get; set; }
    public int GameState { get; set; }
    public bool ReachedFinish { get; set; }
    public bool GameOver { get; set; }
    public bool Completed { get; set; }
    public bool GamePaused { get; set; }
    public bool JumpHeld { get; set; }
    public double TotalTime { get; set; }
    public double TotalTimeMs { get; set; }
    public float SecondsOnThisLevel { get; set; }
    public float ScreenLeftBound { get; set; }
    public float CheckpointPosition { get; set; }
    public int DistanceToTravel { get; set; }
    public PlayerSnapshot? Player { get; set; }
    public List<TrackSnapshot> TracksAhead { get; set; } = new();
    public List<EntitySnapshot> EntitiesAhead { get; set; } = new();
}

internal sealed class PlayerSnapshot
{
    public VectorSnapshot Position { get; set; } = new();
    public VectorSnapshot Velocity { get; set; } = new();
    public BoundsSnapshot? Bounds { get; set; }
    public bool Grounded { get; set; }
    public bool Jumping { get; set; }
    public bool JumpReady { get; set; }
    public string CurrentTrackType { get; set; } = "";
    public int CurrentTrackTypeId { get; set; }
}

internal sealed class TrackSnapshot
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Dx { get; set; }
    public BoundsSnapshot? Bounds { get; set; }
    public string Type { get; set; } = "";
    public int TypeId { get; set; }
    public bool HasObstacle { get; set; }
    public string? ObstacleType { get; set; }
    public BoundsSnapshot? ObstacleBounds { get; set; }
}

internal sealed class EntitySnapshot
{
    public int Id { get; set; }
    public string Type { get; set; } = "";
    public float X { get; set; }
    public float Y { get; set; }
    public float Dx { get; set; }
    public bool Visible { get; set; }
    public bool Enabled { get; set; }
    public bool IsObstacle { get; set; }
    public bool IsPickup { get; set; }
    public BoundsSnapshot? Bounds { get; set; }
}

internal sealed class VectorSnapshot
{
    public float X { get; set; }
    public float Y { get; set; }
}

internal sealed class BoundsSnapshot
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
}
