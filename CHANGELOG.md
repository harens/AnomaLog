# Changelog

## [0.4.0](https://github.com/harens/AnomaLog/compare/v0.3.0...v0.4.0) (2026-05-03)


### Features

* add next-event prediction diagnostics to DeepCASE and DeepLog models ([1b48275](https://github.com/harens/AnomaLog/commit/1b4827599795ff4d5184035132483e6f5c59d758))
* add optional per-event anomaly labels to TemplateSequence and validate alignment ([3820888](https://github.com/harens/AnomaLog/commit/3820888f53306eaa69cf3ea414afe0df9ef08d8d))
* **build_templated_dataset:** coarse lock over dataset name + cache path ([5ab01b5](https://github.com/harens/AnomaLog/commit/5ab01b50b535e82fe2689e628060a5e52c366bc6))
* **deepcase:** add event-level prediction metrics, not just sequence ([436e154](https://github.com/harens/AnomaLog/commit/436e15460e30ef98589af90ad3a07d2ef6dca9a9))
* **deepcase:** enhance training feedback by reporting progress per epoch ([7693743](https://github.com/harens/AnomaLog/commit/76937432241fdac12a6ff7fb3b3e1055a4c2f2eb))
* **deepcase:** update documentation and tests for zero-query iterations in scoring ([c2d0629](https://github.com/harens/AnomaLog/commit/c2d0629b7d2d9cc0c042db8720a4efa824c04ca6))
* **deeplog:** add progress reporting for key model and parameter schema preparation ([a631783](https://github.com/harens/AnomaLog/commit/a631783c8a5a211066da27e497ed352d00fbaa34))
* **detectors:** implement BatchExperimentDetector for bulk scoring and enhance evaluation logic ([b1f0ff8](https://github.com/harens/AnomaLog/commit/b1f0ff8015a2cdf0f23c24d8cd71fe891b89332f))
* **docs:** add reference documentation for experiments package ([80e20ae](https://github.com/harens/AnomaLog/commit/80e20aeb4e6ae439c9cfa71bae3effc05d4c74ab))
* **experiment_logger:** enhance logging with concrete run names ([413f52a](https://github.com/harens/AnomaLog/commit/413f52a09865841a1f2ac638456daf62e986ecd1))
* **experiment_runner:** add --write-predictions option to persist predictions.jsonl ([ef06f46](https://github.com/harens/AnomaLog/commit/ef06f468c112af27a3c9a207717b3400d02771e2))
* **experiments:** add DeepLog and DeepCASE detectors ([7b3745e](https://github.com/harens/AnomaLog/commit/7b3745e9ab4e2daf08ba4ffc792ac1ee505515c8))
* **experiments:** allow for multi-process model sweeps ([46e0a05](https://github.com/harens/AnomaLog/commit/46e0a05e5c40a188f4b8db22f9f24ce109f7a6e6))
* **experiments:** lazy load models for optional extra dependencies ([a58e50c](https://github.com/harens/AnomaLog/commit/a58e50c3d873363aeafb0d32fd0c6992a2b99f55))
* **experiments:** share progress totals and logging ([f0019a6](https://github.com/harens/AnomaLog/commit/f0019a636d75b433804516aed6fd5a1b3be09fbf))
* **models:** add SingleFitMixin for single fit state management in detectors ([9f4247b](https://github.com/harens/AnomaLog/commit/9f4247b10c9280c3e47a4cd0d240202b89b46255))
* **models:** support bounded train progress hints ([2b24a2a](https://github.com/harens/AnomaLog/commit/2b24a2af0edb89068f1d2be3c0cec3637813ab31))
* modify train/test fractions in tandem ([28a21f0](https://github.com/harens/AnomaLog/commit/28a21f0f0cd9202e2a843c72fc10ffb70b0d40bd))
* **naive_bayes:** add type annotations and improve docstrings ([b611c63](https://github.com/harens/AnomaLog/commit/b611c634787c580bd60ab2e255ae8aac8f39c564))
* **parquet:** implement chronological entity grouping and persist entity chronology index ([28d7f48](https://github.com/harens/AnomaLog/commit/28d7f48f0002694871e03209bdec85a562992f84))
* set paper-faithful deepcase iterations, expose next event prediction metrics ([0131ccf](https://github.com/harens/AnomaLog/commit/0131ccf1650a2df4ec17f6cdc39393e5b11a270c))
* **tests:** add fixtures to mirror Prefect API URL in subprocess environments ([71c1c23](https://github.com/harens/AnomaLog/commit/71c1c2384ff392d5bec64c40ff023b3ab1ca0578))
* **torch_runtime:** add shared helpers for managing torch device and seed ([b7c0485](https://github.com/harens/AnomaLog/commit/b7c0485c11b590f087c5c1932e7dfb3373692a1b))


### Bug Fixes

* **deepcase:** don't treat abstained scores as anomalous ([0b99151](https://github.com/harens/AnomaLog/commit/0b99151533918f7f42a7011d325b9b369e43bccb))
* **deepcase:** optimise template access and label resolution in training batch ([9155033](https://github.com/harens/AnomaLog/commit/91550331999deba7156b12424526278ab3912ee3))
* **deepcase:** use finer grained anomaly labels where available ([03741f0](https://github.com/harens/AnomaLog/commit/03741f092806e480892aa671493197217abad397))
* **deeplog:** next-event predictions over all logs, not just latest one ([c20858e](https://github.com/harens/AnomaLog/commit/c20858e21ea56a429e19066bc3f5770a349326f4))
* evaluator no longer treats every DeepCASE outcome as abstained ([9e8a1ef](https://github.com/harens/AnomaLog/commit/9e8a1ef1110934bb0fca5769f3904d8787a54aed))
* **experiments:** correct supervised entity split fractions ([0e27773](https://github.com/harens/AnomaLog/commit/0e27773d1a83c74cb6e52c5963d0d19e2f666133))
* **hdfs_v1:** add support for non-integer csv anomaly labels ([7a2f4df](https://github.com/harens/AnomaLog/commit/7a2f4dffb0595b2028a0f5421e3532e96c94875e))
* keep test set the same across different train splits ([d27fc42](https://github.com/harens/AnomaLog/commit/d27fc424e3513accf4a641992440d282a7f49eba))
* paper-faithful split contract changes ([da2d6c5](https://github.com/harens/AnomaLog/commit/da2d6c5313e312c8319cbe269620c39715af1b3a))
* **tests:** keep deeplog/case runners independent of dataset.toml ([f003c94](https://github.com/harens/AnomaLog/commit/f003c94544e19648e9ff87424e952eb1406cd03b))


### Documentation

* **anomalog:** apply strict pydoclint across modules ([8429e76](https://github.com/harens/AnomaLog/commit/8429e76895f76d660a8268f7b72b20edf3057aa1))
* **experiments:** document caching and reruns ([5240b0d](https://github.com/harens/AnomaLog/commit/5240b0dd1bbe6ec9fa9ed5ad5fb3eb81e22bcf67))
* **experiments:** document DeepLog and DeepCASE support ([8b5bda5](https://github.com/harens/AnomaLog/commit/8b5bda5e3376f4de8512a0956fbaaf682e991597))
* **experiments:** pydoclint docstrings ([8c082df](https://github.com/harens/AnomaLog/commit/8c082df42642138e383430afc5f5ae098892a188))
* **tests:** apply strict pydoclint ([0978c04](https://github.com/harens/AnomaLog/commit/0978c041d0a819f2f7ad31bf7870b839b2f385b1))

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
