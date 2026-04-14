# Changelog

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
