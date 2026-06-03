# Changelog

## [0.5.0](https://github.com/harens/AnomaLog/compare/v0.4.0...v0.5.0) (2026-06-03)


### Features

* **bgl:** add CFDR BGL dataset preset ([a47de4a](https://github.com/harens/AnomaLog/commit/a47de4a532b2f9f31c0c0ee72b5742d0d58766c0))


### Bug Fixes

* **deeplog:** wire continuous context through chronological streams ([fd6a09c](https://github.com/harens/AnomaLog/commit/fd6a09ce7f57537064a5326b3f530ca06df6dda4))

## [0.4.0](https://github.com/harens/AnomaLog/compare/v0.3.0...v0.4.0) (2026-06-03)


### Features

* add next-event prediction diagnostics to DeepCASE and DeepLog models ([1b48275](https://github.com/harens/AnomaLog/commit/1b4827599795ff4d5184035132483e6f5c59d758))
* Add OpenStackDeepLogParser and SpellTemplateParser for OpenStack log processing ([2892c72](https://github.com/harens/AnomaLog/commit/2892c723fa482e7980820d78018797bf456ff109))
* add optional per-event anomaly labels to TemplateSequence and validate alignment ([3820888](https://github.com/harens/AnomaLog/commit/3820888f53306eaa69cf3ea414afe0df9ef08d8d))
* add Slurm job wrappers and configuration management ([5ca223b](https://github.com/harens/AnomaLog/commit/5ca223b8af8a7f7e045fe7fb67d094b988974f51))
* **ait-ads:** suppress sequence-level metrics in reports and add related tests ([a0f6808](https://github.com/harens/AnomaLog/commit/a0f6808910413a16a7cede80a1a8e7ed11f3e8fb))
* **anomalog:** add preprocessed DeepLog session source ([a2aeffc](https://github.com/harens/AnomaLog/commit/a2aeffc0bec739cc73caf38b9c9c3789bd8f42b7))
* **baselines:** run with all ML models ([8638cf2](https://github.com/harens/AnomaLog/commit/8638cf26e3a8ca1a73992313e120a53946c66ac9))
* **bgl:** split CCS 2017 and How Far Are We protocols ([51a2e4f](https://github.com/harens/AnomaLog/commit/51a2e4f3703cef9ca2790a1f01f445e77204d611))
* **build_templated_dataset:** coarse lock over dataset name + cache path ([5ab01b5](https://github.com/harens/AnomaLog/commit/5ab01b50b535e82fe2689e628060a5e52c366bc6))
* **datasets:** add AIT-ADS scenario support ([1a7ec8c](https://github.com/harens/AnomaLog/commit/1a7ec8c70e6f0a2e5fb2e3c74618c82f9fe30be8))
* **deepcase:** add cluster-labelling ablations to registry ([d451235](https://github.com/harens/AnomaLog/commit/d451235ce6cb7fa9e482d33f0e3b3c66a4172bf2))
* **deepcase:** add event-level prediction metrics, not just sequence ([436e154](https://github.com/harens/AnomaLog/commit/436e15460e30ef98589af90ad3a07d2ef6dca9a9))
* **deepcase:** enhance training feedback by reporting progress per epoch ([7693743](https://github.com/harens/AnomaLog/commit/76937432241fdac12a6ff7fb3b3e1055a4c2f2eb))
* **deepcase:** update documentation and tests for zero-query iterations in scoring ([c2d0629](https://github.com/harens/AnomaLog/commit/c2d0629b7d2d9cc0c042db8720a4efa824c04ca6))
* **deeplog:** add HDFS short-session padding variant ([e63086e](https://github.com/harens/AnomaLog/commit/e63086e3cef98ad8a0e1d3b3595dec39e61ac388))
* **deeplog:** add OpenStack parameter CI approximation ([ce38437](https://github.com/harens/AnomaLog/commit/ce3843736c0a9ae43a60d8506784af3df26c4f86))
* **deeplog:** add OpenStack regression coverage ([b2d8470](https://github.com/harens/AnomaLog/commit/b2d8470d948c7b85303c305db83f5129f9a0f565))
* **deeplog:** add progress reporting for key model and parameter schema preparation ([a631783](https://github.com/harens/AnomaLog/commit/a631783c8a5a211066da27e497ed352d00fbaa34))
* **deeplog:** add short-session padding fidelity mode ([e5ada94](https://github.com/harens/AnomaLog/commit/e5ada9439e3cb5f98b3c55ce813131a1ecb48aaf))
* **deeplog:** align BGL 2022 path with Drain3 and CI highlights ([834e308](https://github.com/harens/AnomaLog/commit/834e308e2e9792a2ff7af97f78e6a650536a2ade))
* **deeplog:** carry parameter history across openstack entity windows ([d7d4a13](https://github.com/harens/AnomaLog/commit/d7d4a1341e2aa1f22b7e055eb6dd2601895ebcb1))
* **deeplog:** disable parameter model by default, correct g default value ([fe77e1c](https://github.com/harens/AnomaLog/commit/fe77e1cf7d8417fa7e9bf214db0866ea7a74e598))
* **deeplog:** emit parameter ci report artifacts ([c243b1f](https://github.com/harens/AnomaLog/commit/c243b1f920c47d53a0ac6924b69cc797359334f8))
* **deeplog:** left pad by default ([c41195d](https://github.com/harens/AnomaLog/commit/c41195da933ed0743b2197dc7224b41c6402c48f))
* **deeplog:** make top-g replay configurable ([2b55195](https://github.com/harens/AnomaLog/commit/2b55195fff166d61b9bf5e248632efc74407ecb0))
* **deeplog:** merge continuous context series in parameter dataset construction ([5967617](https://github.com/harens/AnomaLog/commit/5967617e76e2377abe10b654b739d58231019212))
* **deeplog:** separate parameter summary from debug trace ([3556dca](https://github.com/harens/AnomaLog/commit/3556dcacf3496b5fd0941fd63abbf3858b07603b))
* **deeplog:** support parameter-only scoring and registry updates ([ea44148](https://github.com/harens/AnomaLog/commit/ea44148ebf585b8f2c404d9fab33e05b346f97e5))
* **deeplog:** surface OpenStack Figure 9 metadata in parameter reports ([945ae45](https://github.com/harens/AnomaLog/commit/945ae45b7c4f56abf9961120bd10dfcf7f7a8a3c))
* **dependencies:** add spellpy as a git dependency and update related markers ([b0d079c](https://github.com/harens/AnomaLog/commit/b0d079cb114578fbbc794c287842c8e8deade4aa))
* **detectors:** implement BatchExperimentDetector for bulk scoring and enhance evaluation logic ([b1f0ff8](https://github.com/harens/AnomaLog/commit/b1f0ff8015a2cdf0f23c24d8cd71fe891b89332f))
* **docs:** add reference documentation for experiments package ([80e20ae](https://github.com/harens/AnomaLog/commit/80e20aeb4e6ae439c9cfa71bae3effc05d4c74ab))
* Enhance continuous context handling in DeepLog ([e464816](https://github.com/harens/AnomaLog/commit/e4648160a7fa4fe31c9b07b5b79cbbc3723a566c))
* enhance materialization to handle stale Prefect cache paths and update related tests ([9feaa97](https://github.com/harens/AnomaLog/commit/9feaa972aabbad9d0654aa30f024bfc214eaffa4))
* **entity_chronological:** add AIT-ADS entity-chronological dataset and update related configurations ([dcd771a](https://github.com/harens/AnomaLog/commit/dcd771abf0ca9a5825b241c7e3c112a13e8c8b3a))
* **experiment_logger:** enhance logging with concrete run names ([413f52a](https://github.com/harens/AnomaLog/commit/413f52a09865841a1f2ac638456daf62e986ecd1))
* **experiment_runner:** add --write-predictions option to persist predictions.jsonl ([ef06f46](https://github.com/harens/AnomaLog/commit/ef06f468c112af27a3c9a207717b3400d02771e2))
* **experiment_runner:** enhance failure handling to log bundle failures without stopping execution ([a9a97e4](https://github.com/harens/AnomaLog/commit/a9a97e461d86193a1a0326ae1d2fcfbb3941d730))
* **experiments:** add DeepLog and DeepCASE detectors ([7b3745e](https://github.com/harens/AnomaLog/commit/7b3745e9ab4e2daf08ba4ffc792ac1ee505515c8))
* **experiments:** add event-level baseline reporting ([8cf2c13](https://github.com/harens/AnomaLog/commit/8cf2c13d2fa07a923ecea31ba9f54c73c896cc58))
* **experiments:** add HDFS Table-IV compatibility suite ([488bf5e](https://github.com/harens/AnomaLog/commit/488bf5e71b0d387c3d280e3e71896c12526942a1))
* **experiments:** add Prefect telemetry configuration to Slurm wrapper ([cd70adb](https://github.com/harens/AnomaLog/commit/cd70adbb84e0eefea2af96a9b52216982cc05c7b))
* **experiments:** allow for multi-process model sweeps ([46e0a05](https://github.com/harens/AnomaLog/commit/46e0a05e5c40a188f4b8db22f9f24ce109f7a6e6))
* **experiments:** commit registry-backed experiment overhaul ([af97052](https://github.com/harens/AnomaLog/commit/af97052d5a043be2795cf49be5540641685d8364))
* **experiments:** enhance error handling in submit_experiments for sbatch failures ([c0d6e9b](https://github.com/harens/AnomaLog/commit/c0d6e9b2243a9b90fcb58a00f2943e549466eb86))
* **experiments:** lazy load models for optional extra dependencies ([a58e50c](https://github.com/harens/AnomaLog/commit/a58e50c3d873363aeafb0d32fd0c6992a2b99f55))
* **experiments:** make data/cache root configurable for slurm jobs ([caf620e](https://github.com/harens/AnomaLog/commit/caf620e620ccca2a8abef4e0c07735a762e1e177))
* **experiments:** publish scoped metric blocks in run outputs ([1fcae3d](https://github.com/harens/AnomaLog/commit/1fcae3d61a2f04cc5c85ac12e32f79fcbbcbe838))
* **experiments:** separate run_groups to execute ([5759e81](https://github.com/harens/AnomaLog/commit/5759e813277b5b54ed46dca507dcf7afb5521141))
* **experiments:** share progress totals and logging ([f0019a6](https://github.com/harens/AnomaLog/commit/f0019a636d75b433804516aed6fd5a1b3be09fbf))
* **experiments:** streamline run metric metadata ([c8ce126](https://github.com/harens/AnomaLog/commit/c8ce1263220b9802e05de2fec4353fe8637ae9ad))
* **experiments:** submit Slurm arrays in one job ([18bd438](https://github.com/harens/AnomaLog/commit/18bd438fc00075b52436c5ab5697316d753864cd))
* group missing suite reruns by experiment ([0133d2e](https://github.com/harens/AnomaLog/commit/0133d2e5fbf0959e5c1bd722b5510e593c0293fb))
* **markov:** introduce markov baseline ([d53a315](https://github.com/harens/AnomaLog/commit/d53a3150a21578567a06b60f59c76921e8636b45))
* **metadata:** record DeepCASE version in environment metadata ([0250d97](https://github.com/harens/AnomaLog/commit/0250d97673468eaa56e63cd8baf0f9b800dad382))
* **models:** add SingleFitMixin for single fit state management in detectors ([9f4247b](https://github.com/harens/AnomaLog/commit/9f4247b10c9280c3e47a4cd0d240202b89b46255))
* **models:** support bounded train progress hints ([2b24a2a](https://github.com/harens/AnomaLog/commit/2b24a2af0edb89068f1d2be3c0cec3637813ab31))
* modify train/test fractions in tandem ([28a21f0](https://github.com/harens/AnomaLog/commit/28a21f0f0cd9202e2a843c72fc10ffb70b0d40bd))
* **naive_bayes:** add type annotations and improve docstrings ([b611c63](https://github.com/harens/AnomaLog/commit/b611c634787c580bd60ab2e255ae8aac8f39c564))
* **openstack:** make the Figure 9 anomaly slice explicit ([2ca83e0](https://github.com/harens/AnomaLog/commit/2ca83e0c60baee8cd24584b6a29a3b4051132226))
* **parquet:** implement chronological entity grouping and persist entity chronology index ([28d7f48](https://github.com/harens/AnomaLog/commit/28d7f48f0002694871e03209bdec85a562992f84))
* **parsers:** add context manager for spellpy logging and update tests ([16a01f5](https://github.com/harens/AnomaLog/commit/16a01f50259d3e62a1d9d1c4248d443159d6aec2))
* **parsers:** implement lightweight mode for SpellTemplateParser and add corresponding test ([47869ca](https://github.com/harens/AnomaLog/commit/47869caf9931ac8684e369dd8b58744853470e5f))
* **parsing:** add thunderbird parser and timing logs ([f750391](https://github.com/harens/AnomaLog/commit/f7503914f1fa47ca2014d100de91c9898b393f2d))
* **registry:** add HDFS DeepLog Drain3 ablations ([0be1600](https://github.com/harens/AnomaLog/commit/0be1600a9a2cd24214379e1432091f910ccb9150))
* **run_suite:** add check for missing concrete runs and update tests ([6b5e28e](https://github.com/harens/AnomaLog/commit/6b5e28e08aff527eb263b2b40189582e657b7082))
* **run_suite:** enhance output handling for missing runs and update rerun command format ([d824176](https://github.com/harens/AnomaLog/commit/d8241763e30299e4aa5a3b1d49cd6e276fc1c997))
* **sequences:** infer split labels from preprocessed entity prefixes ([c276b7c](https://github.com/harens/AnomaLog/commit/c276b7c39342b82436f3ddec7e8ebf2b377d3abc))
* set paper-faithful deepcase iterations, expose next event prediction metrics ([0131ccf](https://github.com/harens/AnomaLog/commit/0131ccf1650a2df4ec17f6cdc39393e5b11a270c))
* **slurm:** add initial sbatch experiment scripts ([a2aa3ba](https://github.com/harens/AnomaLog/commit/a2aa3baab3b8d79a66849c5fcaf80bf6fde19bdb))
* **slurm:** enhance job scripts to dynamically set repository root and cache directory ([674f476](https://github.com/harens/AnomaLog/commit/674f4763a500d560c050f94e0e7b2186b5fff79c))
* **slurm:** simplify REPO_ROOT assignment in Slurm scripts and tests ([26b0f0d](https://github.com/harens/AnomaLog/commit/26b0f0d6a37b359fe413906e1706b243236a3c66))
* **slurm:** update wrap script to export EXPERIMENT_NAME for nested commands ([d82fa79](https://github.com/harens/AnomaLog/commit/d82fa79be876e620e0116eb8774760877f4651ed))
* **slurm:** update wrap script to use loop for experiment indexing ([f038698](https://github.com/harens/AnomaLog/commit/f038698c933febbd2d19c5f45953330717615141))
* **slurm:** update wrap script to use set -- for experiment indexing ([9b84a8d](https://github.com/harens/AnomaLog/commit/9b84a8ddc5b8c52a3b37a7b873dc5a8e87d04a96))
* **sources:** add preprocessed and thunderbird tests ([c80f5be](https://github.com/harens/AnomaLog/commit/c80f5beed90b26437953c34f1b6c9d6686fcfaf0))
* **sources:** add tar.gz support ([4964c07](https://github.com/harens/AnomaLog/commit/4964c07b2ec895d5748bea30380e49cf13e96af3))
* **template_frequency:** clarify documentation on threshold calibration and update label checks ([ff609ea](https://github.com/harens/AnomaLog/commit/ff609ea9b0ffc7e9fc8ac254aad94df48e862a03))
* **tests:** add fixtures to mirror Prefect API URL in subprocess environments ([71c1c23](https://github.com/harens/AnomaLog/commit/71c1c2384ff392d5bec64c40ff023b3ab1ca0578))
* **tests:** add test for entity counting using grouped rows in ParquetStructuredSink ([236ea9e](https://github.com/harens/AnomaLog/commit/236ea9e16ba915e2bbf905f3a096b658c25d3872))
* **tests:** add test for parquet dataset schema exposing timestamp and partition fields ([bbbc128](https://github.com/harens/AnomaLog/commit/bbbc12819a4754c9780d25bc46f72fba4ea9d2c8))
* **tests:** add unit tests for DeepLog preprocessed dataset source helpers ([60f9be6](https://github.com/harens/AnomaLog/commit/60f9be62cc164b99dd6482554f3278d6ba53fa26))
* **tests:** update wrap script assertions to reflect changes in error handling ([1f55820](https://github.com/harens/AnomaLog/commit/1f55820b4004bc36a9b1524cd03fb990f358fc10))
* **thunderbird:** add entity-grouped DeepCASE extension and update registry ([d3c2561](https://github.com/harens/AnomaLog/commit/d3c2561a93038fe6cd6532bec5ef5392e8e4f4ca))
* **thunderbird:** consolidate registry entries ([3336713](https://github.com/harens/AnomaLog/commit/333671305a9f09c46e6e369b5d5f6183a88e0904))
* **thunderbird:** enhance parser to handle message tails and update entity ID extraction ([a684bdf](https://github.com/harens/AnomaLog/commit/a684bdfa2ee50b8fe90221f1970fd01dcc80410e))
* **thunderbird:** normalise benchmark slice input ([0d96003](https://github.com/harens/AnomaLog/commit/0d960034abb1afc49f6e1cd2b3043ba99ac45ac5))
* **torch_runtime:** add shared helpers for managing torch device and seed ([b7c0485](https://github.com/harens/AnomaLog/commit/b7c0485c11b590f087c5c1932e7dfb3373692a1b))
* update experiment configurations to use chronological datasets and remove obsolete files ([0c34300](https://github.com/harens/AnomaLog/commit/0c34300737bb1460a857063194aa9508821489f3))
* update metrics structure to use canonical metric blocks and remove legacy fields ([7256c05](https://github.com/harens/AnomaLog/commit/7256c0503920583e222aa970039b0f41fe9c971e))


### Bug Fixes

* accept all-normal inline-labelled structured datasets ([dde55eb](https://github.com/harens/AnomaLog/commit/dde55ebf0e33f31597ef976564e80bbd91f356e0))
* **ait-ads:** split by chronology before entity grouping ([7268f8a](https://github.com/harens/AnomaLog/commit/7268f8aba974fb393d354f026b89a1c378f6db6c))
* align BGL 10% DeepLog with normal-entry split ([bc349b3](https://github.com/harens/AnomaLog/commit/bc349b3ac532cc810cfa641b68c6f20f7158a39d))
* align Thunderbird fixed-window contract ([75bffbe](https://github.com/harens/AnomaLog/commit/75bffbe665288b608cd03646ad480256cc3ba06d))
* bump deepcase commit version for batching fix ([1043917](https://github.com/harens/AnomaLog/commit/10439177861a7a460cf874513af51b702e038c43))
* **cache:** enhance result storage handling and cache policy configuration ([5025ef5](https://github.com/harens/AnomaLog/commit/5025ef57632804bd227926d91de2a42b61b2584e))
* **cache:** stop cached materialisation results pointing to stale per-run paths ([fbac953](https://github.com/harens/AnomaLog/commit/fbac953f0fab468d66e67ddcc3277ba5da38491a))
* **deepcase:** align HDFS compat split with entity grouping ([9030c01](https://github.com/harens/AnomaLog/commit/9030c01fe40a8f031f8f1a28ea043756072846a1))
* **deepcase:** align query defaults with paper ([3f5a000](https://github.com/harens/AnomaLog/commit/3f5a000b8263b3ec648f268b2656cf8e4ed701bc))
* **deepcase:** align workload reduction with paper semantics ([9b6b520](https://github.com/harens/AnomaLog/commit/9b6b520832c2db2e7a101f5007dd1ac61cf7749d))
* **deepcase:** cache training chunks and separate query budget ([913bd90](https://github.com/harens/AnomaLog/commit/913bd90104c9394174fc609b9568dacdb5554e00))
* **deepcase:** chunk adapter batches to bound memory ([1a07946](https://github.com/harens/AnomaLog/commit/1a0794677294b410780c7414cae1311c5cdb91bc))
* **deepcase:** don't treat abstained scores as anomalous ([0b99151](https://github.com/harens/AnomaLog/commit/0b99151533918f7f42a7011d325b9b369e43bccb))
* **deepcase:** exclude `evaluation_event_mask` events from score ([c736cd8](https://github.com/harens/AnomaLog/commit/c736cd8890df786523183647a49326fc6b5ca3cd))
* **deepcase:** optimise template access and label resolution in training batch ([9155033](https://github.com/harens/AnomaLog/commit/91550331999deba7156b12424526278ab3912ee3))
* **deepcase:** preserve optimiser state in chunked training ([0e27c5f](https://github.com/harens/AnomaLog/commit/0e27c5f727057a7bb297eaaa2b0dd4b6840fa0c1))
* **deepcase:** reduce fit-time memory pressure ([08352da](https://github.com/harens/AnomaLog/commit/08352da9bae608729713a0bf1b3a233960951696))
* **deepcase:** use finer grained anomaly labels where available ([03741f0](https://github.com/harens/AnomaLog/commit/03741f092806e480892aa671493197217abad397))
* **deeplog:** evaluator no longer repeatedly charging warm-up penalty ([8d9b32f](https://github.com/harens/AnomaLog/commit/8d9b32f11c88dcf92bd6205d457d59dbf56694b7))
* **deeplog:** harden OpenStack Figure 9 parameter CI ([7aa7bae](https://github.com/harens/AnomaLog/commit/7aa7baea4f1b9321d8dcbd7c74ca257a681347f8))
* **deeplog:** keep compatibility variant on deeplog_default ([95df577](https://github.com/harens/AnomaLog/commit/95df577c42747a608fd7c28149ddd188c32f01ff))
* **deeplog:** microbatch key-model training ([6a391c2](https://github.com/harens/AnomaLog/commit/6a391c284bfea169c006de41a5cb492179dba4c9))
* **deeplog:** next-event predictions over all logs, not just latest one ([c20858e](https://github.com/harens/AnomaLog/commit/c20858e21ea56a429e19066bc3f5770a349326f4))
* **deeplog:** remove g=11 from default replay cutoffs ([742bb69](https://github.com/harens/AnomaLog/commit/742bb6905fdc344bd49e0da29543ea71434c9d41))
* **deeplog:** remove short-session padding fallback ([bf94416](https://github.com/harens/AnomaLog/commit/bf9441613908a3d79e97c9016b874866774180de))
* **deeplog:** stream Spell input during training ([7f740a0](https://github.com/harens/AnomaLog/commit/7f740a0d2cb554c1cdd57e19e8133b6a70c4c3f1))
* **deeplog:** use increased epochs/batch size for hdfs pre-processed ([d137477](https://github.com/harens/AnomaLog/commit/d13747782cee8420b497c0924809cb8b26a23752))
* **dependencies:** add diff-cover to dev dependency group ([66f41b4](https://github.com/harens/AnomaLog/commit/66f41b4aa92c0ac5596185e02aa8ab11173d91ab))
* don't consume log sequences between different models ([7a53d32](https://github.com/harens/AnomaLog/commit/7a53d3239aa15628ce7d9caf83d38c4763b396d9))
* drop incompatible baselines from paper configs ([f020967](https://github.com/harens/AnomaLog/commit/f020967c2795ffcaa3cd6d59e4c79e14affe3243))
* drop obsolete spellpy persist_state argument ([e536a43](https://github.com/harens/AnomaLog/commit/e536a439636d2547e6bf67624a45587fee09dfbd))
* evaluator no longer treats every DeepCASE outcome as abstained ([9e8a1ef](https://github.com/harens/AnomaLog/commit/9e8a1ef1110934bb0fca5769f3904d8787a54aed))
* **experiments:** apply deepcase model-set overrides ([64f0f12](https://github.com/harens/AnomaLog/commit/64f0f1208a8cf540033d657e62e9e465e5c6903c))
* **experiments:** correct supervised entity split fractions ([0e27773](https://github.com/harens/AnomaLog/commit/0e27773d1a83c74cb6e52c5963d0d19e2f666133))
* **experiments:** invalidate dataset cache on force reruns ([826abfb](https://github.com/harens/AnomaLog/commit/826abfbb4abc42e8747d06b054e9551399f44620))
* **experiments:** raise Prefect startup timeout for dataset builds ([e38d2e5](https://github.com/harens/AnomaLog/commit/e38d2e543070ea0bcacc5361bbfdba7059da6c74))
* **hdfs_v1:** add support for non-integer csv anomaly labels ([7a2f4df](https://github.com/harens/AnomaLog/commit/7a2f4dffb0595b2028a0f5421e3532e96c94875e))
* keep test set the same across different train splits ([d27fc42](https://github.com/harens/AnomaLog/commit/d27fc424e3513accf4a641992440d282a7f49eba))
* make OpenStack DeepLog preprocessing paper-faithful ([6c37d8d](https://github.com/harens/AnomaLog/commit/6c37d8d06aa4fa587c0c08a12fc69a0c8c9c4110))
* **markov:** accept mixed BGL training chunks ([823c94f](https://github.com/harens/AnomaLog/commit/823c94fd1ce56f537df1bfa01e253d36de12fcc9))
* **markov:** remove quadratic calibration scan ([7638d2b](https://github.com/harens/AnomaLog/commit/7638d2b96fd7524ddc4b86819563c70b8e9ed881))
* **openstack-deeplog:** normalise volatile session tokens ([958e522](https://github.com/harens/AnomaLog/commit/958e522db8a07301bae32128e287d7761e80d23b))
* **openstack:** group by instance id ([c9b75c7](https://github.com/harens/AnomaLog/commit/c9b75c7f7c6abe323a631bde5eca4f0c9b921407))
* paper-faithful split contract changes ([da2d6c5](https://github.com/harens/AnomaLog/commit/da2d6c5313e312c8319cbe269620c39715af1b3a))
* **parquet-sink:** rebuild missing structured cache directories ([19557ca](https://github.com/harens/AnomaLog/commit/19557ca1ccf7b5fa11caf751406099ddb2e43123))
* **parquet:** sort rows within each entity by line order for accurate grouping ([24e4253](https://github.com/harens/AnomaLog/commit/24e425391fcb64e66f2bf39cddfa92e3042fa4d6))
* **parquet:** stale parquet cache making entity scan look empty ([4c85634](https://github.com/harens/AnomaLog/commit/4c8563428c57a460a4598aac5b0cb620a4a149ec))
* **parquet:** validate cached parquet fragments before reuse ([409d915](https://github.com/harens/AnomaLog/commit/409d915e146b808ba2ac1e695a94a67bbf3f9fac))
* **parsers:** fix openstack dropping traceback/error continuation rows ([6407caf](https://github.com/harens/AnomaLog/commit/6407caffd0cd599b29ba7300f072c37140a365a8))
* **parsers:** simplify SpellTemplateParser after spellpy fix ([f383178](https://github.com/harens/AnomaLog/commit/f383178529bdcd55ff23e917c4279813f939009d))
* record file-boundary split provenance in manifests ([56a2c47](https://github.com/harens/AnomaLog/commit/56a2c4736d8d093acf9e9b04622d6f8df2611017))
* recover stale Prefect cache paths ([b01eae2](https://github.com/harens/AnomaLog/commit/b01eae2c0f56c9dc2e5c6d53bcb09d0b0877a6ae))
* repair chronological stream and audit typing ([4d7f94b](https://github.com/harens/AnomaLog/commit/4d7f94be0c442bd28d31043e2a6f31b06c36581e))
* resolve paths in ExperimentBundle for consistent path handling ([029fe01](https://github.com/harens/AnomaLog/commit/029fe01d0976ab86ac0c375ca453f3288760efc7))
* **ruff:** update external linting rules to include only "DOC" ([f68660f](https://github.com/harens/AnomaLog/commit/f68660f5f3782cb998c697d89e50eaf5ec9654b6))
* **run_bundle:** improve error handling for existing result paths ([0a283de](https://github.com/harens/AnomaLog/commit/0a283defd86e1dc1f99e59b354585bd960d3876b))
* **spell:** collapse back to direct spellpy parsing ([3b55bbf](https://github.com/harens/AnomaLog/commit/3b55bbf10f81997e63d12038e888a18de68e7790))
* **template:** clear stale spell cache before retraining ([f142c57](https://github.com/harens/AnomaLog/commit/f142c576ec51b419dbe94dd7aa3df540a39386d6))
* **tests:** keep deeplog/case runners independent of dataset.toml ([f003c94](https://github.com/harens/AnomaLog/commit/f003c94544e19648e9ff87424e952eb1406cd03b))
* **thunderbird:** invalidate template cache by raw slice ([33afd82](https://github.com/harens/AnomaLog/commit/33afd82d34e50df8c3264a12ad6d68785ed51b71))
* **thunderbird:** skip empty Thunderbird message rows ([a225a79](https://github.com/harens/AnomaLog/commit/a225a791fd66b5b614e60c929c880911881564e8))


### Performance Improvements

* cache registry bundle decoding ([fce28e2](https://github.com/harens/AnomaLog/commit/fce28e25a9640f0ad4a68e95d86943e6d5fb4fe4))
* **deeplog:** speed up Thunderbird key training ([3272761](https://github.com/harens/AnomaLog/commit/327276160901f10a08e34059e24eee2311bafa1a))


### Documentation

* **anomalog:** apply strict pydoclint across modules ([8429e76](https://github.com/harens/AnomaLog/commit/8429e76895f76d660a8268f7b72b20edf3057aa1))
* **deeplog:** audit templates ([82eede4](https://github.com/harens/AnomaLog/commit/82eede40241af656f55fa45c97d618f8877eda13))
* **deeplog:** document duplicate-session findings in HDFS preprocessed test files ([74b389a](https://github.com/harens/AnomaLog/commit/74b389a4415c3b65f7733c79b86036e1ba0ea699))
* **experiments:** document caching and reruns ([5240b0d](https://github.com/harens/AnomaLog/commit/5240b0dd1bbe6ec9fa9ed5ad5fb3eb81e22bcf67))
* **experiments:** document DeepLog and DeepCASE support ([8b5bda5](https://github.com/harens/AnomaLog/commit/8b5bda5e3376f4de8512a0956fbaaf682e991597))
* **experiments:** pydoclint docstrings ([8c082df](https://github.com/harens/AnomaLog/commit/8c082df42642138e383430afc5f5ae098892a188))
* **experiments:** refresh Slurm registry examples ([3b73ca0](https://github.com/harens/AnomaLog/commit/3b73ca0b1769b8de7a83a67edb1d4c5166b91903))
* **experiments:** satisfy pydoclint for Slurm backend ([d594306](https://github.com/harens/AnomaLog/commit/d5943064c4f831d8f2ae2ac56fd0ff5d4bcb5e31))
* **tests:** apply strict pydoclint ([0978c04](https://github.com/harens/AnomaLog/commit/0978c041d0a819f2f7ad31bf7870b839b2f385b1))
* **thunderbird:** note cache-key fix and template contract ([541ecc7](https://github.com/harens/AnomaLog/commit/541ecc7054ec6474c1e2689f2637b17eba2a7caf))
* **thunderbird:** record parser skips and DeepLog comparison ([c75c635](https://github.com/harens/AnomaLog/commit/c75c635652a134947d7bce771777a545997e7641))
* **thunderbird:** refresh reproduction notes and config ([a3f3172](https://github.com/harens/AnomaLog/commit/a3f317223fc6a5278ffe28f438af09a267ab2ec9))
* tighten docstrings for pydoclint ([85db1be](https://github.com/harens/AnomaLog/commit/85db1bea245f6240a248e1a43725427100f81c90))

## [0.3.0](https://github.com/harens/AnomaLog/compare/v0.2.0...v0.3.0) (2026-04-14)


### Features

* **experiments:** add config-driven detector runs ([c63ef3d](https://github.com/harens/AnomaLog/commit/c63ef3de1e519765ae1518a27771f3a97dc80e8d))
* **registry:** add named resolvers for built-in presets and parsers ([ae41996](https://github.com/harens/AnomaLog/commit/ae419964dc0b6b7b0be3569bb8381e81ca0f8826))
* **representations:** add model-facing sequence views ([076b6ff](https://github.com/harens/AnomaLog/commit/076b6ffd9dd7c7cd9a3e2330e5e9cffd61387f46))


### Bug Fixes

* **ci:** downgrade Python version to 3.13 for CI jobs ([17a5762](https://github.com/harens/AnomaLog/commit/17a576270e4a7c80a26d41ae51133cb804488ad7))
* **ci:** include all groups including experiments ([399f6ab](https://github.com/harens/AnomaLog/commit/399f6aba251deaa90b57ea8e75b87613de9dd343))
* **ci:** pydoclint docstring-parser error ([9ca4ed8](https://github.com/harens/AnomaLog/commit/9ca4ed84af5a79017fcec2ff104411d852938d12))
* **labels:** treat non-zero anomaly labels as anomalous ([cfc979b](https://github.com/harens/AnomaLog/commit/cfc979bf03a1b506933901d0cf5d356f457c6b21))
* **parquet:** tolerate vanished output dirs during rewrite ([6bfb1ae](https://github.com/harens/AnomaLog/commit/6bfb1ae1a28bb0b91f22335bff207c4b02a42c89))
* **tooling:** scope pre-commit pydoclint ([34e908b](https://github.com/harens/AnomaLog/commit/34e908bde9d36d1d860dd1e286d996f6d33e3814))


### Documentation

* **api:** align docstrings with pydoclint ([086361e](https://github.com/harens/AnomaLog/commit/086361e2c99e798cc6f6a71b51e4a3e1bc7321c3))
* **experiments:** include arg docstrings ([cc00529](https://github.com/harens/AnomaLog/commit/cc00529fd9fdf9b69cc0b536f22ca652bfecb07b))
* restructure documentation with new getting started and development guides, remove outdated quickstart ([c538f48](https://github.com/harens/AnomaLog/commit/c538f484327069ca5cac3fc614e3d43370e8abf5))
* rewrite onboarding around representations and experiments ([e0a7c84](https://github.com/harens/AnomaLog/commit/e0a7c84349985bba334c2bbda7e953919242470b))
* **sequences:** fix docstring for windows ([9547a8f](https://github.com/harens/AnomaLog/commit/9547a8f5acd76f22726b3767f311e83733a96e9e))
* **tests:** include arg docstrings ([bece668](https://github.com/harens/AnomaLog/commit/bece668eaedfd77c1497570ea67b9f603133c587))

## [0.2.0](https://github.com/harens/AnomaLog/compare/v0.1.0...v0.2.0) (2026-03-31)


### ⚠ BREAKING CHANGES

* documentation now targets DatasetSpec and anomalog.presets instead of the previous RawDataset and anomalog.datasets examples
* **api:** replaced RawDataset and anomalog.datasets entrypoints with DatasetSpec(...).from_source(...).parse_with(...).label_with(...).template_with(...).build(); moved anomaly label readers to anomalog.labels; custom DatasetSource implementations must implement raw_logs_path/raw_logs_relpath and custom template parsers must accept dataset_name=... at runtime
* **parsers:** moved structured and template parser imports from anomalog.structured_parsers and anomalog.template_parsers to anomalog.parsers.*

### Features

* **api:** add DatasetSpec builder and dataset presets ([bab9df9](https://github.com/harens/AnomaLog/commit/bab9df92119ef47204a0ac5e8c70c386d9a41678))


### Bug Fixes

* **cache:** rerun materialized work when local artifacts are missing ([60635ae](https://github.com/harens/AnomaLog/commit/60635aef9ce5a0fac6f4c9d94a65c5fd9056d4af))
* **tests:** gitignore include integration log file ([8355a90](https://github.com/harens/AnomaLog/commit/8355a902231176b7d8fd1100f69437268d161737))


### Documentation

* rewrite README and quickstart for fluent DatasetSpec API ([76a7b4e](https://github.com/harens/AnomaLog/commit/76a7b4e1e525c5327631294b681447d56eef77a2))


### Miscellaneous Chores

* release 0.2.0 ([b4df838](https://github.com/harens/AnomaLog/commit/b4df838bef05dffb469a5eda6a7fc0104f309fd7))


### Code Refactoring

* **parsers:** move parser modules under anomalog.parsers ([9ced56d](https://github.com/harens/AnomaLog/commit/9ced56d30fc0192cae0104664d00c2e22bba9457))

## 0.1.0 (2026-03-27)


### Features

* **all:** introduce prefect, refactor structure, parquet writer ([7250d63](https://github.com/harens/AnomaLog/commit/7250d6398601c55ada7610436496f13f7a8bc4ac))
* **anomaly_label_reader:** implement CSV and inline label readers ([190f376](https://github.com/harens/AnomaLog/commit/190f3769e4acc9b0b94ee606a3d90ca3e08c4d2c))
* **cache:** cache class arguments by source ([f2fdb19](https://github.com/harens/AnomaLog/commit/f2fdb19da433d4e99c68984937d9c39d3eef4727))
* **cache:** only cache on direct file deps, not parents ([edb75d2](https://github.com/harens/AnomaLog/commit/edb75d227fcb84d2df92e306661d24692116bbf2))
* **cache:** update CACHE_POLICY configuration ([5aa01cb](https://github.com/harens/AnomaLog/commit/5aa01cb8e6ec20a798bc6a3f7c7f47dcc99ba4ba))
* **contracts:** update count_entities method to count entities by label ([360ed9a](https://github.com/harens/AnomaLog/commit/360ed9a7deee84adfac6a3e17a54216ea3141e0b))
* **datasets:** begin fluent API design ([d4813a4](https://github.com/harens/AnomaLog/commit/d4813a454b9f0e2b327b3e28b491354cb262c2f8))
* **datasets:** implement Drain3 parser ([c886bb9](https://github.com/harens/AnomaLog/commit/c886bb9a9f60ccd86325d45d74e707ded46674f6))
* determine whether anomalies are inline ([9174c50](https://github.com/harens/AnomaLog/commit/9174c50a631807cc983678d18fa811c595eb1493))
* **fetch_data:** add BGL details ([1aa9f20](https://github.com/harens/AnomaLog/commit/1aa9f20afc93cc850ff5dc9f99b17fc4d2f58220))
* **fetch_data:** add zip file integrity check before extraction ([d84db67](https://github.com/harens/AnomaLog/commit/d84db673da8b7dc83e38478f51d6e52e97f476e8))
* **fetch_data:** fetch HDFS dataset, handle cancel download, add progress bars ([1b99c67](https://github.com/harens/AnomaLog/commit/1b99c67e7878728891ebb31a3ad49ff91b8c96a0))
* **fetch_data:** remove dataset zip after download, remove extra wrapper directory ([3bb4bf8](https://github.com/harens/AnomaLog/commit/3bb4bf801b8d1b7de0b3d02fc5a5573837d2646a))
* **fetch_data:** switch from loghub-2 -&gt; loghub, note paths to logs/labels ([3860d39](https://github.com/harens/AnomaLog/commit/3860d394e702543eb620a84ee289aca293e8b89f))
* **label_reader:** speed up line/group label reading ([24161c0](https://github.com/harens/AnomaLog/commit/24161c00b8e5e1b8fd4694f0e295c472c05bce92))
* **main:** define flow for hdfs v1 ([7e3d0ad](https://github.com/harens/AnomaLog/commit/7e3d0adc82319f96bba2a3783676e16602ae78ed))
* **models:** fix splitting leakage ([4886b01](https://github.com/harens/AnomaLog/commit/4886b010c857b2c476790f364def19c25dca5fe6))
* **models:** refactor pre/post-model steps into separate classes ([9dd6b45](https://github.com/harens/AnomaLog/commit/9dd6b45a7134a204592612478ac5506cba390aa9))
* **NaiveBayes:** initial version ([675c2d7](https://github.com/harens/AnomaLog/commit/675c2d76c3f572f555f7db84a72e75d786ca000d))
* **parquet:** add explicit schema and preserve write order for structured batches ([56db413](https://github.com/harens/AnomaLog/commit/56db4134bd2f3eada9e7b7aac428d2782910c5d2))
* **ParquetStructuredSink:** hive partitioning and improved performance ([ac9e536](https://github.com/harens/AnomaLog/commit/ac9e53629142370a385353c94a177e8b5852897d))
* **raw_dataset:** log example unstructured line content ([ddb8098](https://github.com/harens/AnomaLog/commit/ddb8098870b574088ab87b220efd4acd49e6f1d5))
* **SequenceBuilder:** add option to train on normal entities only ([fef6ca3](https://github.com/harens/AnomaLog/commit/fef6ca3937f42d54ce4070dda64c481b38e78f8d))
* **SequenceBuilder:** optimixe label retrieval with caching for improved performance ([dab24bd](https://github.com/harens/AnomaLog/commit/dab24bdeb2c22cb94e9b4c7bfaaca84e13e766a0))
* **Sequences:** Support entity/time grouping of logs ([923410e](https://github.com/harens/AnomaLog/commit/923410e79b58e9b94e76b55c2dd7093fe96ab091))
* **structured_parsers:** implement hdfs v1 parser ([365a1b4](https://github.com/harens/AnomaLog/commit/365a1b4fdb0eba4fc66ce1dcd2d1c316e2ea235d))
* **StructuredDataset:** refactor template mining to use structured lines ([379ac7d](https://github.com/harens/AnomaLog/commit/379ac7db8106fa9af4da71e00d6b414990018d73))
* **templated_dataset:** implement GroupedDatasetView for entity, fixed, and time windows ([683e2b8](https://github.com/harens/AnomaLog/commit/683e2b8acbc400299db3a524b56419cdc74f14ff))
* **writer_worker:** switch to hive partitioning on entities ([9c86453](https://github.com/harens/AnomaLog/commit/9c86453baac6b5a48913d2050f1cb59ea3ad7799))


### Bug Fixes

* **cache/classes:** hash class instances ([ac5ff23](https://github.com/harens/AnomaLog/commit/ac5ff230b000eba8d063112d9ee8e0be3dae7783))
* **cache:** fix AssetDepsFingerprintPolicy for template mining ([ffd7b91](https://github.com/harens/AnomaLog/commit/ffd7b911564b8ea79eaa4d886b46f45793d25055))
* **docs:** add edit URI for documentation editing ([6aa0951](https://github.com/harens/AnomaLog/commit/6aa095117ebc52f185324107bf444c6201c1d05c))
* **drain3_parser:** define inference function creation even with cache loading ([7bb2fa0](https://github.com/harens/AnomaLog/commit/7bb2fa00e02be7923450383a8eb3174f42b71b30))
* **io_utils, local:** ensure destination directory is not created on invalid zips ([3a472eb](https://github.com/harens/AnomaLog/commit/3a472ebe770ea63addbf611bc7484d41f873e8fd))
* **ParquetStructuredSink:** fix structured line processing with time window streaming ([1460661](https://github.com/harens/AnomaLog/commit/146066141a172a5b974c89e98331191fbfbeccb8))
* **remote_zip:** invalidate cache if download doesn't exist ([64e41a4](https://github.com/harens/AnomaLog/commit/64e41a45c0fafb1f9cdfa7ec64aea466d557b504))
* **remote_zip:** set correct extraction directory ([8b3c6f5](https://github.com/harens/AnomaLog/commit/8b3c6f5123ae3f8f834ada40546805f60bf7a52e))
* **sequences:** enforce window size and time span for grouping modes ([9706878](https://github.com/harens/AnomaLog/commit/9706878b04dd1d1aaf082371fe9461f92e3d10cc))


### Documentation

* add Sphinx documentation setup and templates ([19fb0f8](https://github.com/harens/AnomaLog/commit/19fb0f81a021b7fe57182bd19bb21c2c81906f5f))
* **cache_class_key_fn:** enhance docstring with examples and clarifications ([3fd8d02](https://github.com/harens/AnomaLog/commit/3fd8d021c8203c4a475f35bbd0b609402256729c))
* enhance documentation across AnomaLog modules ([8aaec69](https://github.com/harens/AnomaLog/commit/8aaec6919332a79cede01043844db96895e3894f))
* **pyproject.toml:** update package name to AnomaLog ([175d192](https://github.com/harens/AnomaLog/commit/175d19277eb589f8c760c07d6c113d4eae1be524))
* **README:** add initial README with project overview and key features ([753950c](https://github.com/harens/AnomaLog/commit/753950c0930bfbca6d925698ee3a58920c8d4882))
* **README:** add research usage details ([5c9d963](https://github.com/harens/AnomaLog/commit/5c9d96325069b41e19a21e4494eb518b9e2bd815))
* transition from sphinx to zensical ([02e3a11](https://github.com/harens/AnomaLog/commit/02e3a1126db3ea58e7d97a8ab6232f1fe4f99099))
* update README with additional badges and formatting improvements ([e2e3c12](https://github.com/harens/AnomaLog/commit/e2e3c1223e28dcf59aa0f5f2728eee8eb51a1863))
