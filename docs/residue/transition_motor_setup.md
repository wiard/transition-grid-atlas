# Transition Motor Setup

`Transition Motor: H_BMW_TPT(theta)`  
Doel: korte, stabiele burn duration zonder extra fase-ruis  
Primaire ordeparameter: `CA50_coherence`  
Secundaire ordeparameters: `CA10-90`, `sigma_CA50`, `COV_IMEP`, `theta_pmax`, `PRR_max`, `knock_margin`, `exergy_loss`

## Hamiltoniaan-configuratie

```yaml
H_BMW_TPT:
  fixed_grid:
    hardware_geometry: immutable
    injector_location: immutable
    spark_location: immutable
    valve_train: immutable
    turbo_layout: immutable
  allowed_knobs:
    spark_timing:
      role: phase_anchor
      residue_rule: move CA50 toward exergy-optimal phase, not toward max pressure alone
    injection_pressure:
      role: mixture_wave_sharpener
      residue_rule: increase atomization only while sigma_CA50 and wall-wet residue decline
    start_of_injection:
      role: mixture_preparation_phase
      residue_rule: avoid late-rich pockets and avoid early wall-film residue
    injection_split:
      role: local_equivalence_smoothing
      residue_rule: split only when it lowers CA10-90 and COV_IMEP together
    intake_VANOS:
      role: trapped_mass_phase_gate
      residue_rule: tune charge motion and residual fraction without increasing pumping residue
    exhaust_VANOS:
      role: residual_entropy_gate
      residue_rule: retain useful thermal residue, reject dilution that increases misfire variance
    Valvetronic_lift:
      role: throttle_loss_suppression
      residue_rule: control load by valve lift while preserving tumble and stable flame initiation
    boost_setpoint:
      role: density_potential
      residue_rule: raise charge density only inside knock, PRR, and heat-loss bounds
    lambda:
      role: chemical_phase_density
      residue_rule: avoid mixtures that shorten burn but increase NOx/knock/noise residue
```

De kern van de configuratie is: Valvetronic bepaalt de luchtmassa met minimale pompverliezen, VANOS bepaalt de interne fasepositie van charge en residu, injectie bepaalt de mengselcoherentie, ontsteking bepaalt CA50. BMW's recente zescilinder TwinPower Turbo-configuraties combineren onder meer twin-scroll turbocharging, High Precision Injection, Valvetronic en Double VANOS; bij de B58 werd voor High Precision Injection een druk tot 5.076 psi genoemd.

## Schone burn-duration residue

```yaml
burn_duration_policy:
  minimize:
    - CA10_90
    - CA0_10_ignition_delay_variance
    - sigma_CA50
    - COV_IMEP
    - heat_transfer_residue
    - knock_correction_events
  reject_if:
    - CA10_90 decreases but sigma_CA50 increases
    - pressure_rise_rate approaches NVH/knock boundary
    - spark_advance creates negative-work residue before TDC
    - boost increase creates disproportionate heat-loss residue
    - injection pressure improves power but worsens wall-film residue
```

## Residue-map

```yaml
clean_configuration:
  phase_target:
    CA50: calibrated_MBT_exergy_window
    theta_pmax: calibrated_after_TDC_window
    note: CA50 is not treated as universal; it is engine/load/speed dependent
  wave_shape:
    CA0_10: stable_kernel_formation
    CA10_50: rapid_but_not_knock_limited
    CA50_90: complete_without_late_heat_release_tail
  actuator_priority:
    1: Valvetronic_lift_for_load_without_throttle_loss
    2: VANOS_for_trapped_mass_and_residual_phase
    3: injection_timing_pressure_for_homogeneous_fast_kernel
    4: spark_timing_for_CA50_lock
    5: boost_only_inside_knock_and_PRR_margin
```

CA50 en piekdrukfase zijn geschikte fasecoordinaten, maar geen vaste absolute waarheden: onderzoek naar verbrandingsfasering laat zien dat optimale CA50 en piekdrukhoek afhangen van ontwerp- en bedrijfsvariabelen, vooral warmteoverdracht, compressieverhouding en burn duration. Voor conventionele configuraties werd in die studie een CA50-gebied rond enkele graden tot circa 11 graden na TDC gevonden, maar de residue gebruikt dit alleen als startvenster, niet als dogma.
