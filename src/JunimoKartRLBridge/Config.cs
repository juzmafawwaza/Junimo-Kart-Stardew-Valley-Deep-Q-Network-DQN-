namespace JunimoKartRLBridge;

internal sealed class Config
{
    public string BindAddress { get; set; } = "127.0.0.1";
    public int Port { get; set; } = 8765;
    public bool StartServerOnLaunch { get; set; } = true;
    public int MaxTracks { get; set; } = 24;
    public int MaxEntities { get; set; } = 24;
    public float LookaheadPixels { get; set; } = 1600f;
    public float LookbehindPixels { get; set; } = 128f;
    public bool AutoAdvanceTitleAfterStart { get; set; } = true;
    public int AutoAdvanceTitleTicks { get; set; } = 180;
    public bool AutoContinueProgressModeNonGameplayStates { get; set; } = true;
    public bool ForceRunWhenUnfocused { get; set; } = true;
}
