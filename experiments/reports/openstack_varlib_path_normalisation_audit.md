# OpenStack `/var/lib/nova/instances` Path Normalisation Audit

This note records a bounded audit of the special OpenStack path collapse that
maps `/var/lib/nova/instances/...` to `INSTANCE_PATH`.

The goal was to check whether removing only that rule materially changes the
template inventory or the separation between normal and abnormal sessions.
The rest of the OpenStack parser remained unchanged:

- UUIDs, IPs, generic path tokens, hex strings, and standalone numbers were
  still normalised.
- The parser still grouped by recovered `instance_id`.
- The only change in the variant was to leave `/var/lib/nova/instances/...`
  visible after the generic path normalisation step.

## Method

I compared the current parser against a temporary subclass that disabled only
the `INSTANCE_PATH` substitution.

For each parser variant, I:

- parsed `openstack_normal1.log` as `openstack_train`
- parsed `openstack_normal2.log` as `openstack_test_normal`
- parsed `openstack_abnormal.log` as `openstack_test_abnormal`
- trained Spell on the train split only
- inferred templates for the train and both test splits

This mirrors the file-boundary OpenStack preprocessing contract without making
any permanent code changes.

## Retained Rows

The `/var/lib` rule does not change parseability.

| Split | Retained rows, current | Retained rows, variant |
| --- | ---: | ---: |
| train (`openstack_normal1.log`) | 14,421 | 14,421 |
| test normal (`openstack_normal2.log`) | 34,097 | 34,097 |
| test abnormal (`openstack_abnormal.log`) | 5,100 | 5,100 |

## Template Inventory

| Split | Templates, current | Templates, variant |
| --- | ---: | ---: |
| train | 19 | 574 |
| test normal | 20 | 1,331 |
| test abnormal | 18 | 213 |

### Overlap with train vocabulary

| Comparison | Current | Variant |
| --- | ---: | ---: |
| train vs test normal | 19 | 18 |
| train vs test abnormal | 18 | 17 |

### Abnormal-only templates

| Variant | Unique templates only seen in abnormal |
| --- | ---: |
| current | 0 |
| variant | 196 |

### Unknown targets under the train vocabulary

| Split | Current | Variant |
| --- | ---: | ---: |
| test normal | 1 | 1,313 |
| test abnormal | 0 | 196 |

The current parser has one unseen normal-test template:

- `During sync_power_state the instance has a pending task (networking). Skip.`

The variant turns almost the entire abnormal split into unseen path-specific
deletion templates. That is not a useful anomaly signal by itself.

## Top Affected Templates

The rule change is dominated by one high-frequency swap and then a long tail of
one-off path-specific deletion templates.

| Template | Current count | Variant count | Delta |
| --- | ---: | ---: | ---: |
| `Deleting instance files INSTANCE_PATH` | 2,064 | 0 | -2,064 |
| `Deletion of <*> complete` | 0 | 2,064 | +2,064 |
| `Deletion of INSTANCE_PATH complete` | 2,064 | 0 | -2,064 |
| `Deleting instance files /var/lib/nova/instances/002afded-8d0b-49f2-bab7-07e8ff79eac4_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/0039c90c-b94c-4a97-985c-0421663a49fc_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/00533c6c-c5fd-NUM-ad06-508ea82a2c2e_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/0060e3e6-bfb7-4eaf-937b-30c303472315_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/007ddfb6-fcdb-4e6c-a8ef-9c3b9fa27b64_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/009c6a16-a3d3-4bbc-9a2e-ba70a49793cc_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/00a09545-b546-45fb-bc9a-b227fdf4b036_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/00d0361d-326e-473b-NUM-05506438830b_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/00dec35d-985a-40f6-NUM-6938a2eaff7e_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/00fde18e-312b-NUM-NUM-b8eb7fee155c_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/01083cc5-ec46-4fac-803f-02fdb1b5b408_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/012438a0-b548-4e1f-bf17-b2fd8e1e3b4d_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/012de84d-befe-4e95-9abd-394f908d7d2d_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/012f2a88-7b8d-NUM-981f-8df6b3b6aee3_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/015f9c70-78bc-4f68-b9f7-7fe8e6b07ff1_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/016e3623-3b17-463b-91e0-dfb7fd10178e_del` | 0 | 1 | +1 |
| `Deleting instance files /var/lib/nova/instances/018e5eb5-d5e8-NUM-ab9d-536d415a90ba_del` | 0 | 1 | +1 |

## Conclusion

Removing the special `INSTANCE_PATH` rule does **not** look like a defensible
improvement.

- It massively increases vocabulary size.
- It creates 196 abnormal-only templates.
- It turns almost all test targets into unknowns under the train vocabulary.
- The new templates are overwhelmingly path-specific `/var/lib/nova/instances`
  deletion artefacts, not stable behavioural distinctions.

So the rule is more than a cosmetic normalisation step, but it is still best
understood as hygiene that suppresses filesystem-noise variants rather than as
a source of paper-faithful anomaly signal.
