# Supply Transition Motor Setup

```yaml
H_Supply:
  fixed_grid:
    warehouse_capacity: immutable
    physical_layout: immutable
    supplier_lead_time_contract: immutable
    sku_topology: immutable
    review_calendar: immutable
  allowed_knobs:
    service_level_target:
      role: service_phase_anchor
      residue_rule: raise only if carrying entropy remains bounded
    review_period:
      role: observation_cadence
      residue_rule: shorten only if ordering noise does not increase
    reorder_point:
      role: collapse_threshold
      residue_rule: move only when forecast error and lead time variance justify it
    safety_stock_ratio:
      role: uncertainty_buffer
      residue_rule: absorb variance without creating chronic overstock
    order_up_to_level:
      role: target_inventory_band
      residue_rule: restore steady state without exceeding capacity
    batch_size_aggregation:
      role: capital_wave_packet
      residue_rule: aggregate only while service coherence and recovery improve
    emergency_replenishment_threshold:
      role: irreversibility_boundary
      residue_rule: emergency orders must remain exceptional
```

```yaml
clean_inventory_policy:
  minimize:
    - stockout_rate
    - carrying_cost_per_unit_service
    - recovery_time_after_spike
    - lead_time_variance_residue
    - emergency_order_rate
    - capacity_violation_count
    - theta_drift
  reject_if:
    - service_level_improves_but_carrying_cost_explodes
    - stockout_cost_drops_but_overstock_becomes_structural
    - recovery_depends_on_emergency_orders
    - capacity_is_violated
    - reorder_points_are_manually_tuned_without_registry_update
```

De Transition Motor is geen solverimplementatie, maar een governance-specificatie voor hoe `theta` als auditeerbare voorraadconfiguratie wordt behandeld.
