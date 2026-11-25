# Export each slide of manuscript/Paper6_Figures_revised.pptx to figures/v12/figN.png (3200 px wide).
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "manuscript\Paper6_Figures_revised.pptx"
$out = Join-Path $root "figures\v12"
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($src, $true, $false, $false)
for ($i = 1; $i -le $p.Slides.Count; $i++) { $p.Slides.Item($i).Export((Join-Path $out "fig$i.png"), "PNG", 3200, 1800) }
$p.Close()
