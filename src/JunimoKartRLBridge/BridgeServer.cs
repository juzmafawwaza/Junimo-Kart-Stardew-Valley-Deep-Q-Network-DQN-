using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using StardewModdingAPI;

namespace JunimoKartRLBridge;

internal sealed class BridgeServer : IDisposable
{
    private readonly ModEntry mod;
    private readonly IMonitor monitor;
    private readonly Config config;
    private TcpListener? listener;
    private Thread? listenThread;
    private volatile bool running;

    public BridgeServer(ModEntry mod, IMonitor monitor, Config config)
    {
        this.mod = mod;
        this.monitor = monitor;
        this.config = config;
    }

    public void Start()
    {
        if (this.running)
            return;

        var endpoint = new IPEndPoint(IPAddress.Parse(this.config.BindAddress), this.config.Port);
        this.listener = new TcpListener(endpoint);
        this.listener.Start();
        this.running = true;
        this.listenThread = new Thread(this.ListenLoop)
        {
            IsBackground = true,
            Name = "JunimoKartRLBridge"
        };
        this.listenThread.Start();
        this.monitor.Log($"Junimo Kart RL bridge listening on {this.config.BindAddress}:{this.config.Port}.", LogLevel.Info);
    }

    public void Dispose()
    {
        this.running = false;
        try
        {
            this.listener?.Stop();
        }
        catch
        {
            // ignored during shutdown
        }
    }

    private void ListenLoop()
    {
        while (this.running)
        {
            try
            {
                using var client = this.listener!.AcceptTcpClient();
                this.HandleClient(client);
            }
            catch (SocketException)
            {
                if (this.running)
                    this.monitor.Log("Socket error in Junimo Kart RL bridge listener.", LogLevel.Warn);
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            catch (Exception ex)
            {
                this.monitor.Log($"Unhandled bridge listener error: {ex}", LogLevel.Error);
            }
        }
    }

    private void HandleClient(TcpClient client)
    {
        using var stream = client.GetStream();
        using var reader = new StreamReader(stream);
        using var writer = new StreamWriter(stream) { AutoFlush = true };

        while (this.running && client.Connected)
        {
            var line = reader.ReadLine();
            if (line is null)
                return;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            BridgeResponse response;
            try
            {
                var request = JsonSerializer.Deserialize<ClientRequest>(line, Protocol.JsonOptions);
                response = this.mod.HandleBridgeRequest(request);
            }
            catch (Exception ex)
            {
                response = BridgeResponse.Error(ex.Message, this.mod.GetLatestSnapshot());
            }

            writer.WriteLine(JsonSerializer.Serialize(response, Protocol.JsonOptions));
        }
    }
}
