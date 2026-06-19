# Espera a que el servidor local de Fuser acepte conexiones y abre la UI
# en su propia ventana (modo "app" de Chrome/Edge: sin barra del navegador).
$ErrorActionPreference = 'SilentlyContinue'

$targetHost = '127.0.0.1'
$port       = 7860
$url        = "http://$targetHost`:$port"

# Esperar hasta ~90 s a que el puerto este escuchando.
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect($targetHost, $port)
        $client.Close()
        $ready = $true
        break
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) { exit 0 }

# Margen para que Gradio termine de montar las rutas.
Start-Sleep -Milliseconds 1000

# Buscar un navegador Chromium para abrir en modo "app" (ventana propia).
$candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)
$browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($browser) {
    Start-Process $browser -ArgumentList "--app=$url"
} else {
    # Sin Chrome/Edge: abrir en el navegador por defecto (pestana normal).
    Start-Process $url
}
