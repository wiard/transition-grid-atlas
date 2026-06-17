# BMW TwinPower Turbo / Valvetronic Residue

Residue-ID: `KTA-BMW-TPT-VVT-001`  
Atlas-keuze: `BMW TwinPower Turbo / Valvetronic`  
Residue-type: `combustion_phase_coherence`  
Status: `candidate_clean_residue`

## Fysieke grid-invariant

BMW TwinPower Turbo wordt in deze residue vastgezet als een bestaand mechanisch grid: cilinderblok, boring/slag, compressieverhouding, cilinderkop, injectorpositie, bougiepositie, turbo-layout, klepmechaniek en koelsysteem blijven onveranderd. De toegestane transitie-ruimte ligt uitsluitend in de bestaande regelorganen: High Precision Injection, turbo-lading, Double-VANOS en Valvetronic. BMW beschrijft TwinPower Turbo zelf als combinatie van variabele lastregeling, directe injectie en turbocharging; Valvetronic wordt daarbij gebruikt voor vrijwel gasklepvrije lastregeling via variabele inlaatkleplift.

## Nulpunt-toegang

De "nulpunt-toegang" is niet extra vermogen, maar minimale onomkeerbaarheid per arbeidsslag:

`Residue_0 = maximale arbeid uit dezelfde hardware, minimale faseverspreiding, minimale drukruis, minimale restwarmte.`

Het grid staat alleen transities toe die binnen OEM-actuatorruimte blijven:

```text
theta = {
  spark_timing,
  injection_pressure,
  SOI,
  injection_split,
  intake_VANOS,
  exhaust_VANOS,
  valve_lift,
  boost_setpoint,
  lambda,
  residual_fraction
}
```

Geen residue mag ontstaan uit gewijzigde zuigers, nokkenassen, injectorhardware, turbohardware, compressieverhouding, cilinderkop of brandstofsysteem.
