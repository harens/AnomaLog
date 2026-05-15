# Experiment Results Bundle

Total JSON files: 32



---

## `bgl_deeplog_paper_10pct_entry_stream_no_online_deeplog_hidden_size_256_history_size_3_num_layers_1_primary_metric_scope_event_level_detection_secondary_metric_scopes_next_event_prediction_sequence_level_detection_top_g_values_1_3_6/0a0045b18533/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "f2dad1558cd56f27ada15ccc4ca411d8f6112c39f4d6b77b16736435ac2f56e1",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_deeplog_paper_10pct_entry_stream_no_online",
  "evaluation_unit": "continuous_event_stream",
  "label_counts": {
    "test": {
      "1": 44
    },
    "train": {
      "1": 4
    }
  },
  "label_unit": "event",
  "model_manifest": {
    "batch_size": 2048,
    "detector": "deeplog",
    "epochs": 300,
    "gaussian_confidence": 0.99,
    "hidden_size": 256,
    "history_size": 3,
    "ignored_sequence_count": 0,
    "implementation_scope": "Scoped DeepLog core v1",
    "include_elapsed_time": true,
    "learning_rate": 0.001,
    "num_layers": 1,
    "parameter_detection_enabled": true,
    "parameter_models": [
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.019075800600482938,
        "gaussian_stddev": 0.013196567301772348,
        "gaussian_upper_bound": 0.05306790536264339,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "NULL DISCOVERY <:*:> Node card VPD check: <:*:> node in processor card slot <:*:> do not match. VPD ecid <:SEQ:>, found <:SEQ:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17,
          18,
          19,
          20,
          21,
          22,
          23,
          24,
          25,
          26
        ],
        "feature_count": 5,
        "feature_names": [
          "dt_prev_ms",
          "param_27",
          "param_28",
          "param_29",
          "param_30"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 1.4639408685760644,
        "gaussian_stddev": 2.0184364014515306,
        "gaussian_upper_bound": 6.663088498784708,
        "input_feature_count": 5,
        "numeric_parameter_positions": [
          27,
          28,
          29,
          30
        ],
        "template": "NULL DISCOVERY ERROR Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> OK. MPGOOD ERROR LATCH IS ACTIVE. The <:NUM:>.<:NUM:> volt rail is OK. The <:NUM:>.<:NUM:> volt rail is OK."
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.6050674154607454,
        "gaussian_stddev": 0.7589417424672362,
        "gaussian_upper_bound": 2.559971795394315,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "NULL DISCOVERY SEVERE Can not get assembly information for node card"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.05328752529374707,
        "gaussian_stddev": 0.053208285085569684,
        "gaussian_upper_bound": 0.19034298520874138,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "NULL DISCOVERY WARNING Node card is not fully functional"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0025139691557322288,
        "gaussian_mean": 0.002771552086087119,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.003029135016442009,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS <:*:> <:*:> ciod: <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 4.3756046300884614e-07,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.00025802049081789883,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS <:*:> <:*:> ciod: Error <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> No such file or directory"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 4.2727626813211746e-08,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.0002576256579817032,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: Error loading /home/yates/BGLa SWL tests/BGLa SWL MPI tests work/BGLa SWL MPI functests/MPITs func tests/doe mpits/test cases/allgather: invalid or missing program image, No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 7.289969788234131e-06,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.00026487290014312414,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: LOGIN <:*:> <:*:> <:*:> failed: No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.023519548343493547,
        "gaussian_stddev": 0.29583727703727836,
        "gaussian_upper_bound": 0.7855458756182292,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: LOGIN <:*:> <:*:> failed: No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.020212911943575863,
        "gaussian_stddev": 0.061131914904737406,
        "gaussian_upper_bound": 0.17767828973725622,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: LOGIN <:*:> <:*:> failed: Permission denied"
      },
      {
        "dropped_parameter_positions": [
          0
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 5.218450454837107e-06,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.0002628013808097271,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: LOGIN <:*:> failed: No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.5323655984945114,
        "gaussian_stddev": 53.63639769334158,
        "gaussian_upper_bound": 138.6905705138064,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0
        ],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_1",
          "param_2"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 3.7361722372907744,
        "gaussian_stddev": 15.67902300591277,
        "gaussian_upper_bound": 44.12265914693824,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          1,
          2
        ],
        "template": "RAS KERNEL <:*:> floating pt ex mode <:NUM:> enable......<:NUM:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.042177230554728336,
        "gaussian_stddev": 0.14199925928611104,
        "gaussian_upper_bound": 0.4079430837061314,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL <:*:> guaranteed <:*:> cache block <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 3.9669404800054835,
        "gaussian_mean": 3.9671980629358385,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 3.9674556458661936,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL FATAL <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.00028593380204108496,
        "gaussian_stddev": 0.008136724682769481,
        "gaussian_upper_bound": 0.02124474767482834,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.3147866512974846,
        "gaussian_stddev": 1.108883224740576,
        "gaussian_upper_bound": 3.171080555798061,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 4.290905200446998,
        "gaussian_mean": 4.291162783377353,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 4.291420366307708,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL FATAL <:*:> error <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          0,
          1
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_2"
        ],
        "gaussian_lower_bound": 1.6925477192060876,
        "gaussian_mean": 1.6928053021364424,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 1.6930628850667973,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          2
        ],
        "template": "RAS KERNEL FATAL capture first <:*:> <:*:> error address..<:NUM:>"
      },
      {
        "dropped_parameter_positions": [
          1,
          2,
          3
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 6.880714499215753e-05,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.00032639007534704753,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          1,
          2,
          3
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.35418557597073663,
        "gaussian_stddev": 1.1058399543887087,
        "gaussian_upper_bound": 3.2026405355203518,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO <:NUM:> <:*:> <:*:> error(s) (dcr <:HEX:>) detected and corrected"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 5,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1",
          "param_2",
          "param_3"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.9483282819474729,
        "gaussian_stddev": 1.1437004706877592,
        "gaussian_upper_bound": 3.894305468827673,
        "input_feature_count": 5,
        "numeric_parameter_positions": [
          0,
          1,
          2,
          3
        ],
        "template": "RAS KERNEL INFO <:NUM:> ddr errors(s) detected and corrected on rank <:NUM:>, symbol <:NUM:>, bit <:NUM:>"
      },
      {
        "dropped_parameter_positions": [
          1,
          2
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 5.604780725917288,
        "gaussian_stddev": 34.08968041540509,
        "gaussian_upper_bound": 93.41397848853477,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO <:NUM:> torus receiver <:*:> input pipe error(s) (dcr <:HEX:>) detected and corrected"
      },
      {
        "dropped_parameter_positions": [
          2
        ],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 2.6814121171534273,
        "gaussian_stddev": 2.6245682265839703,
        "gaussian_upper_bound": 9.441851864351786,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          0,
          1
        ],
        "template": "RAS KERNEL INFO <:NUM:> tree receiver <:NUM:> in re-synch state event(s) (dcr <:HEX:>) detected"
      },
      {
        "dropped_parameter_positions": [
          1,
          2
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.5469724417934297,
        "gaussian_stddev": 1.2978618311139156,
        "gaussian_upper_bound": 3.8900429783342867,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO CE sym <:NUM:>, at <:HEX:>, mask <:HEX:>"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 4,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1",
          "param_2"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 2.1197487345779407,
        "gaussian_stddev": 17.703745678321013,
        "gaussian_upper_bound": 47.7215756353744,
        "input_feature_count": 4,
        "numeric_parameter_positions": [
          0,
          1,
          2
        ],
        "template": "RAS KERNEL INFO ciod: Message code <:NUM:> is not <:NUM:> or <:NUM:>"
      },
      {
        "dropped_parameter_positions": [
          2
        ],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.4484416185294604,
        "gaussian_stddev": 0.7114695748485791,
        "gaussian_upper_bound": 2.2810657980079077,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          0,
          1
        ],
        "template": "RAS KERNEL INFO ciod: cpu <:NUM:> at treeaddr <:NUM:> sent unrecognized message <:HEX:>"
      },
      {
        "dropped_parameter_positions": [
          1,
          2,
          3,
          4,
          5,
          6
        ],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_7"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 57.849637014914656,
        "gaussian_stddev": 324.92962270260585,
        "gaussian_upper_bound": 894.8128807633747,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          0,
          7
        ],
        "template": "RAS KERNEL INFO ciod: for node <:NUM:>, <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.05052554414830838,
        "gaussian_stddev": 0.029648198958662282,
        "gaussian_upper_bound": 0.12689424382347866,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL INFO ciod: pollControlDescriptors: Detected the debugger died."
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 1.2324492683050507,
        "gaussian_stddev": 0.8124655494509687,
        "gaussian_upper_bound": 3.325221838704814,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          0,
          1
        ],
        "template": "RAS KERNEL INFO ddr: activating redundant bit steering: rank=<:NUM:> symbol=<:NUM:>"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 4.453141915437437,
        "gaussian_mean": 4.4533994983677925,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 4.4536570812981475,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL INFO ddr: excessive soft failures, consider replacing the card"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 24.15401309553521,
        "gaussian_stddev": 669.2074337860266,
        "gaussian_upper_bound": 1747.9181311943428,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL INFO instruction cache parity error corrected"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.5598792131656937,
        "gaussian_stddev": 1.6564300545807156,
        "gaussian_upper_bound": 4.8265602870338045,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO total of <:NUM:> ddr error(s) detected and corrected"
      },
      {
        "dropped_parameter_positions": [
          0
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_1"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 22.0137687386999,
        "gaussian_stddev": 49.034902214575496,
        "gaussian_upper_bound": 148.3193067596583,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          1
        ],
        "template": "RAS LINKCARD INFO MidplaneSwitchController performing bit sparing on <:*:> bit <:NUM:>"
      }
    ],
    "parameter_schema_policy": "strict: include only template parameter positions that are always numeric in normal training data",
    "parameter_validation_policy": "per-template temporal tail split over history-target pairs; Gaussian residuals come from held-out validation pairs scored after training on each series prefix",
    "scored_parameter_event_count": 3003350,
    "scored_parameter_event_fraction": 0.7028395339661506,
    "skipped_parameter_model_count": 31,
    "skipped_parameter_models": [
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL CMCS INFO Controlling BG/L rows [ <:NUM:> <:NUM:> <:NUM:> <:NUM:> ]"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL CMCS INFO Running as background command"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL CMCS INFO Starting SystemController"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL HARDWARE WARNING <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> as part of Service Action <:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL HARDWARE WARNING EndServiceAction <:NUM:> performed upon <:*:> by <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "NULL HARDWARE WARNING PrepareForService is being done on this <:*:> mCardSernum(<:SEQ:>), <:*:> mIp(<:IP:>), mType(<:NUM:>)) by root"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to <:IP:>:<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:> <:*:> <:HEX:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL <:*:> error detected in EDRAM bank <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL DDR <:*:> <:*:> <:*:> <:HEX:> <:HEX:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL L3 <:*:> <:*:> register: <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL close EDRAM pages as soon as possible....<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL disable all access to cache directory....<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL disable flagging of DDR UE's as major internal error.<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL machine check interrupt (bit=<:HEX:>): <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL machine check interrupt (bit=<:HEX:>): L2 dcache unit data parity error"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL max number of outstanding prefetches.....<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL memory manager / command manager address parity..<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL memory manager <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL number of correctable errors detected in L3 <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL number of lines with parity errors written to L3 EDRAMs...<:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL parity error in <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL prefetch depth for <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL rts panic! - stopping execution"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL size of <:*:> <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO <:*:> correctable errors exceeds threshold (iar <:HEX:> lr <:HEX:>)"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO <:NUM:> torus <:*:> <:*:> <:*:> error(s) (dcr <:HEX:>) detected and corrected"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO ciod: sendMsgToDebugger: error sending PROGRAM EXITED message to debugger."
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO ciodb has been restarted."
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO idoproxydb has been started: $Name: <:*:> <:NUM:> $ Input parameters: -enableflush -loguserinfo db.properties BlueGene1"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO mmcs db server has been started: <:*:> db server --useDatabase BGL --dbproperties <:*:> --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all"
      }
    ],
    "test_label_counts": {
      "1": 44
    },
    "test_sequence_count": 44,
    "top_g": 6,
    "top_g_values": [
      1,
      3,
      6
    ],
    "train_key_vocabulary_size": 65,
    "train_label_counts": {
      "1": 4
    },
    "train_parameter_covered_event_count": 281881,
    "train_parameter_covered_event_fraction": 0.9997552757581132,
    "train_sequence_count": 4,
    "trained_parameter_model_count": 34,
    "validation_fraction": 0.1
  },
  "prediction_unit": "event",
  "primary_metric_scope": "event_level_detection",
  "raw_entry_split_summary": {
    "application_order": "before_grouping",
    "cutoff_entry_index": 474797,
    "ignored_anomalous_entry_count": 0,
    "ignored_normal_entry_count": 0,
    "ignored_raw_entry_count": 0,
    "split_mode": "raw_entry_prefix_fraction",
    "straddling_group_count": 1,
    "straddling_group_policy": "split_partial_sequences",
    "test_anomalous_entry_count": 155613,
    "test_normal_entry_count": 4117553,
    "test_raw_entry_count": 4273166,
    "train_anomalous_entry_count": 192847,
    "train_normal_entry_count": 281950,
    "train_raw_entry_count": 474797
  },
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "0a0045b1853340714325ca405c32bc3b58ea184765a996c7d1f3008af9b55548",
  "sequence_config": {
    "chunk_size": 100000,
    "grouping": "chronological_stream",
    "split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_entry_fraction": 0.1
    },
    "step": null,
    "test_fraction": 0.9,
    "train_fraction": 0.1
  },
  "sequence_count": 48,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 44,
    "train": 4
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.08333333,
    "eligible_train_sequence_count": 4,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 4,
    "requested_test_fraction": 0.9,
    "requested_train_fraction": 0.1,
    "train_pool_sequence_count": 4
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": "before_grouping",
    "raw_entry_split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_entry_fraction": 0.1
    },
    "raw_entry_split_summary": {
      "application_order": "before_grouping",
      "cutoff_entry_index": 474797,
      "ignored_anomalous_entry_count": 0,
      "ignored_normal_entry_count": 0,
      "ignored_raw_entry_count": 0,
      "split_mode": "raw_entry_prefix_fraction",
      "straddling_group_count": 1,
      "straddling_group_policy": "split_partial_sequences",
      "test_anomalous_entry_count": 155613,
      "test_normal_entry_count": 4117553,
      "test_raw_entry_count": 4273166,
      "train_anomalous_entry_count": 192847,
      "train_normal_entry_count": 281950,
      "train_raw_entry_count": 474797
    },
    "straddling_group_policy": "split_partial_sequences",
    "test_fraction": 0.9,
    "train_fraction": 0.1,
    "train_on_normal_entities_only": null
  },
  "stream_segment_policy": {
    "chunk_size": 100000,
    "mode": "continuous_event_stream"
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_deeplog_paper_10pct_entry_stream_no_online_deeplog_hidden_size_256_history_size_3_num_layers_1_primary_metric_scope_event_level_detection_secondary_metric_scopes_next_event_prediction_sequence_level_detection_top_g_values_1_3_6/0a0045b18533/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "evaluation_unit": "continuous_event_stream",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "event",
  "mean_test_score": 243892586721.21658,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 155613,
        "normal": 4117553
      },
      "confusion_matrix": {
        "fn": 28290,
        "fp": 1918985,
        "tn": 2198568,
        "tp": 127323
      },
      "counted_predictions": 4273166,
      "diagnostics": {
        "events_eligible": 4273166,
        "events_seen": 4273166,
        "source": "event_level_detection"
      },
      "evaluation_unit_count": 4273166,
      "headline_metrics": {
        "accuracy": 0.54430158,
        "f1": 0.1156472,
        "precision": 0.06222084,
        "recall": 0.81820285
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "valid"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.6766378839483418,
          "f1": 0.11443217708342229,
          "precision": 0.13117565726971728,
          "recall": 0.11147997595836821
        },
        "classification_top1_weighted": {
          "accuracy": 0.6766378839483418,
          "f1": 0.6789757395047121,
          "precision": 0.6844391609214928,
          "recall": 0.6766378839483418
        },
        "exclusions": {
          "insufficient_history": 0,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": {
          "expected_insufficient_history_from_segments": 0,
          "history_size": 3,
          "largest_segments": [
            {
              "boundary_reason": "stream_start",
              "insufficient_history": 0,
              "length": 4273166,
              "segment_id": 1,
              "source_object_type": "chronological_chunk"
            }
          ],
          "segment_count": 1,
          "segment_length_histogram": {
            "4273166": 1
          },
          "smallest_segments": [
            {
              "boundary_reason": "stream_start",
              "insufficient_history": 0,
              "length": 4273166,
              "segment_id": 1,
              "source_object_type": "chronological_chunk"
            }
          ]
        },
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.6766378839483418,
            "3": 0.6990039703582777,
            "6": 0.7036246661140709
          },
          "hit_count": {
            "1": 2891386,
            "3": 2986960,
            "6": 3006705
          },
          "k_values": [
            1,
            3,
            6
          ]
        },
        "totals": {
          "coverage": 1.0,
          "events_eligible": 4273166,
          "events_seen": 4273166
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 1.0
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 44,
        "normal": 0
      },
      "confusion_matrix": {
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "tp": 44
      },
      "counted_predictions": 44,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 44,
          "normal": 0
        }
      },
      "evaluation_unit_count": 44,
      "headline_metrics": {},
      "ignored_prediction_count": 0,
      "invalid_reason": "single_class_test_set",
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "invalid"
    }
  },
  "prediction_unit": "event",
  "primary_metric_scope": "event_level_detection",
  "sequence_count": 48,
  "split_policy": {
    "application_order": "before_grouping",
    "raw_entry_split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_entry_fraction": 0.1
    },
    "raw_entry_split_summary": {
      "application_order": "before_grouping",
      "cutoff_entry_index": 474797,
      "ignored_anomalous_entry_count": 0,
      "ignored_normal_entry_count": 0,
      "ignored_raw_entry_count": 0,
      "split_mode": "raw_entry_prefix_fraction",
      "straddling_group_count": 1,
      "straddling_group_policy": "split_partial_sequences",
      "test_anomalous_entry_count": 155613,
      "test_normal_entry_count": 4117553,
      "test_raw_entry_count": 4273166,
      "train_anomalous_entry_count": 192847,
      "train_normal_entry_count": 281950,
      "train_raw_entry_count": 474797
    },
    "straddling_group_policy": "split_partial_sequences",
    "test_fraction": 0.9,
    "train_fraction": 0.1,
    "train_on_normal_entities_only": null
  },
  "stream_segment_policy": {
    "chunk_size": 100000,
    "mode": "continuous_event_stream"
  },
  "test_label_counts": {
    "1": 44
  },
  "test_sequence_count": 44,
  "train_label_counts": {
    "1": 4
  },
  "train_sequence_count": 4
}
```


---

## `bgl_deeplog_paper_1pct_normal_entry_stream_no_online_deeplog_hidden_size_256_history_size_3_num_layers_1_primary_metric_scope_event_level_detection_secondary_metric_scopes_next_event_prediction_sequence_level_detection_top_g_values_1_3_6/e5b54c4d61f7/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "c81ce35a0c9f2c28fe28e9fdbbf88c6b2c88d87bfbb256f48a0bc51e65aaab6b",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_deeplog_paper_1pct_normal_entry_stream_no_online",
  "evaluation_unit": "continuous_event_stream",
  "label_counts": {
    "test": {
      "1": 48
    },
    "train": {}
  },
  "label_unit": "event",
  "model_manifest": {
    "batch_size": 2048,
    "detector": "deeplog",
    "epochs": 300,
    "gaussian_confidence": 0.99,
    "hidden_size": 256,
    "history_size": 3,
    "ignored_sequence_count": 0,
    "implementation_scope": "Scoped DeepLog core v1",
    "include_elapsed_time": true,
    "learning_rate": 0.001,
    "num_layers": 1,
    "parameter_detection_enabled": true,
    "parameter_models": [
      {
        "dropped_parameter_positions": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.016058927017062196,
        "gaussian_mean": 0.017385859351443288,
        "gaussian_stddev": 0.0005151476196628731,
        "gaussian_upper_bound": 0.01871279168582438,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS <:*:> <:*:> ciod: <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          0,
          1,
          2
        ],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 7.312228833858666e-06,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.0002648951591887486,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS APP FATAL ciod: LOGIN <:*:> <:*:> <:*:> failed: No such file or directory"
      },
      {
        "dropped_parameter_positions": [
          1,
          2,
          3
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 2.4645740431466884e-05,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.0002822286707863569,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "dropped_parameter_positions": [
          1,
          2
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_0"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.5336668117686653,
        "gaussian_stddev": 0.5074156298544492,
        "gaussian_upper_bound": 1.8406828602264778,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          0
        ],
        "template": "RAS KERNEL INFO CE sym <:NUM:>, at <:HEX:>, mask <:HEX:>"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 3,
        "feature_names": [
          "dt_prev_ms",
          "param_0",
          "param_1"
        ],
        "gaussian_lower_bound": 0.490329753044191,
        "gaussian_mean": 0.4905873359745459,
        "gaussian_stddev": 0.0001,
        "gaussian_upper_bound": 0.4908449189049008,
        "input_feature_count": 3,
        "numeric_parameter_positions": [
          0,
          1
        ],
        "template": "RAS KERNEL INFO ddr: activating redundant bit steering: rank=<:NUM:> symbol=<:NUM:>"
      },
      {
        "dropped_parameter_positions": [],
        "feature_count": 1,
        "feature_names": [
          "dt_prev_ms"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 7317405.403965598,
        "gaussian_stddev": 160851441.42070293,
        "gaussian_upper_bound": 421643261.7334915,
        "input_feature_count": 1,
        "numeric_parameter_positions": [],
        "template": "RAS KERNEL INFO instruction cache parity error corrected"
      },
      {
        "dropped_parameter_positions": [
          0
        ],
        "feature_count": 2,
        "feature_names": [
          "dt_prev_ms",
          "param_1"
        ],
        "gaussian_lower_bound": 0.0,
        "gaussian_mean": 0.005447095433802786,
        "gaussian_stddev": 0.004945995922920325,
        "gaussian_upper_bound": 0.018187136667294344,
        "input_feature_count": 2,
        "numeric_parameter_positions": [
          1
        ],
        "template": "RAS LINKCARD INFO MidplaneSwitchController performing bit sparing on <:*:> bit <:NUM:>"
      }
    ],
    "parameter_schema_policy": "strict: include only template parameter positions that are always numeric in normal training data",
    "parameter_validation_policy": "per-template temporal tail split over history-target pairs; Gaussian residuals come from held-out validation pairs scored after training on each series prefix",
    "scored_parameter_event_count": 1059858,
    "scored_parameter_event_fraction": 0.2254228041462084,
    "skipped_parameter_model_count": 20,
    "skipped_parameter_models": [
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS APP FATAL ciod: LOGIN <:*:> <:*:> failed: No such file or directory"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS APP FATAL ciod: LOGIN <:*:> failed: No such file or directory"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to <:IP:>:<:NUM:>"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL <:*:> floating pt ex mode <:NUM:> enable......<:NUM:>"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL <:*:> guaranteed <:*:> cache block <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL <:*:>"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:> <:*:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL <:*:> <:*:> <:*:> <:*:> <:HEX:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL FATAL rts panic! - stopping execution"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL INFO <:NUM:> <:*:> <:*:> error(s) (dcr <:HEX:>) detected and corrected"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL INFO <:NUM:> ddr errors(s) detected and corrected on rank <:NUM:>, symbol <:NUM:>, bit <:NUM:>"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO <:NUM:> torus receiver <:*:> input pipe error(s) (dcr <:HEX:>) detected and corrected"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO <:NUM:> tree receiver <:NUM:> in re-synch state event(s) (dcr <:HEX:>) detected"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS KERNEL INFO ddr: excessive soft failures, consider replacing the card"
      },
      {
        "reason": "not enough validation examples for Gaussian calibration",
        "template": "RAS KERNEL INFO total of <:NUM:> ddr error(s) detected and corrected"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO ciodb has been restarted."
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO idoproxydb has been started: $Name: <:*:> <:NUM:> $ Input parameters: -enableflush -loguserinfo db.properties BlueGene1"
      },
      {
        "reason": "not enough training examples after validation split",
        "template": "RAS MMCS INFO mmcs db server has been started: <:*:> db server --useDatabase BGL --dbproperties <:*:> --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all"
      }
    ],
    "test_label_counts": {
      "1": 48
    },
    "test_sequence_count": 48,
    "top_g": 6,
    "top_g_values": [
      1,
      3,
      6
    ],
    "train_key_vocabulary_size": 27,
    "train_label_counts": {},
    "train_parameter_covered_event_count": 11137,
    "train_parameter_covered_event_fraction": 0.2531366487862533,
    "train_sequence_count": 0,
    "trained_parameter_model_count": 7,
    "validation_fraction": 0.1
  },
  "prediction_unit": "event",
  "primary_metric_scope": "event_level_detection",
  "raw_entry_split_summary": {
    "application_order": "before_grouping",
    "cutoff_entry_index": 46318,
    "ignored_anomalous_entry_count": 2322,
    "ignored_normal_entry_count": 0,
    "ignored_raw_entry_count": 2322,
    "split_mode": "raw_entry_prefix_normal_fraction",
    "straddling_group_count": 1,
    "straddling_group_policy": "split_partial_sequences",
    "test_anomalous_entry_count": 346138,
    "test_normal_entry_count": 4355507,
    "test_raw_entry_count": 4701645,
    "train_anomalous_entry_count": 0,
    "train_normal_entry_count": 43996,
    "train_raw_entry_count": 43996
  },
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "e5b54c4d61f7bce22432f1c105734e1985f55483db317e08db47081a5d0eeb0b",
  "sequence_config": {
    "chunk_size": 100000,
    "grouping": "chronological_stream",
    "split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_normal_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_normal_entry_fraction": 0.01
    },
    "step": null,
    "test_fraction": 0.99,
    "train_fraction": 0.01
  },
  "sequence_count": 48,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 48,
    "train": 0
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 0.0,
    "effective_train_fraction_overall": 0.0,
    "eligible_train_sequence_count": 0,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 0,
    "requested_test_fraction": 0.99,
    "requested_train_fraction": 0.01,
    "train_pool_sequence_count": 0
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": "before_grouping",
    "raw_entry_split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_normal_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_normal_entry_fraction": 0.01
    },
    "raw_entry_split_summary": {
      "application_order": "before_grouping",
      "cutoff_entry_index": 46318,
      "ignored_anomalous_entry_count": 2322,
      "ignored_normal_entry_count": 0,
      "ignored_raw_entry_count": 2322,
      "split_mode": "raw_entry_prefix_normal_fraction",
      "straddling_group_count": 1,
      "straddling_group_policy": "split_partial_sequences",
      "test_anomalous_entry_count": 346138,
      "test_normal_entry_count": 4355507,
      "test_raw_entry_count": 4701645,
      "train_anomalous_entry_count": 0,
      "train_normal_entry_count": 43996,
      "train_raw_entry_count": 43996
    },
    "straddling_group_policy": "split_partial_sequences",
    "test_fraction": 0.99,
    "train_fraction": 0.01,
    "train_on_normal_entities_only": null
  },
  "stream_segment_policy": {
    "chunk_size": 100000,
    "mode": "continuous_event_stream"
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_deeplog_paper_1pct_normal_entry_stream_no_online_deeplog_hidden_size_256_history_size_3_num_layers_1_primary_metric_scope_event_level_detection_secondary_metric_scopes_next_event_prediction_sequence_level_detection_top_g_values_1_3_6/e5b54c4d61f7/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "evaluation_unit": "continuous_event_stream",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "event",
  "mean_test_score": 266810602927.08704,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 346138,
        "normal": 4355507
      },
      "confusion_matrix": {
        "fn": 217057,
        "fp": 2001289,
        "tn": 2354218,
        "tp": 129081
      },
      "counted_predictions": 4701645,
      "diagnostics": {
        "events_eligible": 4701645,
        "events_seen": 4701645,
        "source": "event_level_detection"
      },
      "evaluation_unit_count": 4701645,
      "headline_metrics": {
        "accuracy": 0.52817663,
        "f1": 0.10424436,
        "precision": 0.06059088,
        "recall": 0.37291774
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "valid"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.687205435544368,
          "f1": 0.03808950908139438,
          "precision": 0.043841386763586586,
          "recall": 0.039248610718934665
        },
        "classification_top1_weighted": {
          "accuracy": 0.687205435544368,
          "f1": 0.6888247310825505,
          "precision": 0.6917796136464271,
          "recall": 0.687205435544368
        },
        "exclusions": {
          "insufficient_history": 0,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": {
          "expected_insufficient_history_from_segments": 0,
          "history_size": 3,
          "largest_segments": [
            {
              "boundary_reason": "stream_start",
              "insufficient_history": 0,
              "length": 4701645,
              "segment_id": 1,
              "source_object_type": "chronological_chunk"
            }
          ],
          "segment_count": 1,
          "segment_length_histogram": {
            "4701645": 1
          },
          "smallest_segments": [
            {
              "boundary_reason": "stream_start",
              "insufficient_history": 0,
              "length": 4701645,
              "segment_id": 1,
              "source_object_type": "chronological_chunk"
            }
          ]
        },
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.687205435544368,
            "3": 0.7048558536427144,
            "6": 0.7085071288878679
          },
          "hit_count": {
            "1": 3230996,
            "3": 3313982,
            "6": 3331149
          },
          "k_values": [
            1,
            3,
            6
          ]
        },
        "totals": {
          "coverage": 1.0,
          "events_eligible": 4701645,
          "events_seen": 4701645
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 1.0
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 48,
        "normal": 0
      },
      "confusion_matrix": {
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "tp": 48
      },
      "counted_predictions": 48,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 48,
          "normal": 0
        }
      },
      "evaluation_unit_count": 48,
      "headline_metrics": {},
      "ignored_prediction_count": 0,
      "invalid_reason": "single_class_test_set",
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "invalid"
    }
  },
  "prediction_unit": "event",
  "primary_metric_scope": "event_level_detection",
  "sequence_count": 48,
  "split_policy": {
    "application_order": "before_grouping",
    "raw_entry_split": {
      "application_order": "before_grouping",
      "mode": "raw_entry_prefix_normal_fraction",
      "straddling_group_policy": "split_partial_sequences",
      "train_normal_entry_fraction": 0.01
    },
    "raw_entry_split_summary": {
      "application_order": "before_grouping",
      "cutoff_entry_index": 46318,
      "ignored_anomalous_entry_count": 2322,
      "ignored_normal_entry_count": 0,
      "ignored_raw_entry_count": 2322,
      "split_mode": "raw_entry_prefix_normal_fraction",
      "straddling_group_count": 1,
      "straddling_group_policy": "split_partial_sequences",
      "test_anomalous_entry_count": 346138,
      "test_normal_entry_count": 4355507,
      "test_raw_entry_count": 4701645,
      "train_anomalous_entry_count": 0,
      "train_normal_entry_count": 43996,
      "train_raw_entry_count": 43996
    },
    "straddling_group_policy": "split_partial_sequences",
    "test_fraction": 0.99,
    "train_fraction": 0.01,
    "train_on_normal_entities_only": null
  },
  "stream_segment_policy": {
    "chunk_size": 100000,
    "mode": "continuous_event_stream"
  },
  "test_label_counts": {
    "1": 48
  },
  "test_sequence_count": 48,
  "train_label_counts": {},
  "train_sequence_count": 0
}
```


---

## `bgl_entity_chronological_deepcase/3830d9e2d441/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction",
    "manual_workload_reduction",
    "semi_automatic_workload_reduction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "0c5fe4c35909cbd2671fdb4f4cc52e736410e58c165b453c7d8eaa9d4f3825c9",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 29647,
      "1": 25755
    },
    "train": {
      "0": 8230,
      "1": 5620
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "batch_size": 128,
    "cluster_score_strategy": "max",
    "clustered_sample_count": 2007586,
    "confidence_threshold": 0.2,
    "context_length": 10,
    "detector": "deepcase",
    "device": "cuda",
    "epochs": 100,
    "eps": 0.1,
    "hidden_size": 128,
    "ignored_sequence_count": 0,
    "implementation_scope": "Official DeepCase library integration",
    "iterations": 100,
    "known_benign_cluster_count": 1534744,
    "known_cluster_count": 769,
    "known_malicious_cluster_count": 472842,
    "label_policy": "event-label supervision when available: each event-centered sample uses its target event label and falls back to the parent TemplateSequence label when the event label is missing",
    "label_smoothing_delta": 0.1,
    "learning_rate": 0.01,
    "min_samples": 5,
    "no_score": -1,
    "online_updates_status": "not implemented",
    "persistent_cluster_database_status": "not implemented",
    "prediction_diagnostics": {
      "abstained_anomalous_label_count": 5914,
      "abstained_event_count": 822644,
      "abstained_normal_label_count": 28858,
      "confident_anomaly_event_count": 174780,
      "confident_event_count": 1737725,
      "event_count": 2560369,
      "event_decision_metrics": {
        "event_abstain_rate": 0.32129900026129043,
        "event_abstained_decision_count": 822644,
        "event_accuracy": 0.9375436274439282,
        "event_auto_coverage": 0.6787009997387096,
        "event_auto_decision_count": 1737725,
        "event_count": 2560369,
        "event_f1": 0.5527072806851246,
        "event_fn": 807,
        "event_fp": 107725,
        "event_precision": 0.3836537361254148,
        "event_predicted_anomalous_count": 174780,
        "event_predicted_normal_count": 1562945,
        "event_recall": 0.9881082196221744,
        "event_tn": 1562138,
        "event_tp": 67055,
        "event_true_anomalous_count": 127275,
        "event_true_normal_count": 2433094
      },
      "reason_counts": {
        "closest_cluster_outside_epsilon": 285628,
        "known_benign_cluster": 1562945,
        "known_malicious_cluster": 174780,
        "not_confident_enough": 537016
      },
      "sequence_abstained_count": 34772,
      "sequence_confident_anomaly_count": 20630,
      "sequence_confident_normal_count": 0
    },
    "query_batch_size": 1024,
    "random_seed": 0,
    "teach_ratio": 0.5,
    "test_label_counts": {
      "0": 29647,
      "1": 25755
    },
    "test_sequence_count": 55402,
    "timeout_seconds": 86400.0,
    "train_event_vocabulary_size": 168,
    "train_label_counts": {
      "0": 8230,
      "1": 5620
    },
    "train_sample_count": 2187594,
    "train_sequence_count": 13850,
    "unknown_cluster_score_count": 180008
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "3830d9e2d44116fd4b50bf2442ebc4c7c44befd4309143bc011dd77ce4a55b01",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 69252,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 55402,
    "train": 13850
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999422,
    "eligible_train_sequence_count": 13850,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 13850,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 13850
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_entity_chronological_deepcase/3830d9e2d441/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction",
    "manual_workload_reduction",
    "semi_automatic_workload_reduction"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.37236923,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 822644,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 127275,
        "normal": 2433094
      },
      "confusion_matrix": {
        "fn": 807,
        "fp": 107725,
        "tn": 1562138,
        "tp": 67055
      },
      "counted_predictions": 1737725,
      "diagnostics": {
        "event_abstain_rate": 0.32129900026129043,
        "event_auto_coverage": 0.6787009997387096,
        "event_decision_metrics": {
          "event_abstain_rate": 0.32129900026129043,
          "event_abstained_decision_count": 822644,
          "event_accuracy": 0.9375436274439282,
          "event_auto_coverage": 0.6787009997387096,
          "event_auto_decision_count": 1737725,
          "event_count": 2560369,
          "event_f1": 0.5527072806851246,
          "event_fn": 807,
          "event_fp": 107725,
          "event_precision": 0.3836537361254148,
          "event_predicted_anomalous_count": 174780,
          "event_predicted_normal_count": 1562945,
          "event_recall": 0.9881082196221744,
          "event_tn": 1562138,
          "event_tp": 67055,
          "event_true_anomalous_count": 127275,
          "event_true_normal_count": 2433094
        },
        "events_eligible": 1737725,
        "events_seen": 2560369,
        "prediction_diagnostics": {
          "abstained_anomalous_label_count": 5914,
          "abstained_event_count": 822644,
          "abstained_normal_label_count": 28858,
          "confident_anomaly_event_count": 174780,
          "confident_event_count": 1737725,
          "event_count": 2560369,
          "event_decision_metrics": {
            "event_abstain_rate": 0.32129900026129043,
            "event_abstained_decision_count": 822644,
            "event_accuracy": 0.9375436274439282,
            "event_auto_coverage": 0.6787009997387096,
            "event_auto_decision_count": 1737725,
            "event_count": 2560369,
            "event_f1": 0.5527072806851246,
            "event_fn": 807,
            "event_fp": 107725,
            "event_precision": 0.3836537361254148,
            "event_predicted_anomalous_count": 174780,
            "event_predicted_normal_count": 1562945,
            "event_recall": 0.9881082196221744,
            "event_tn": 1562138,
            "event_tp": 67055,
            "event_true_anomalous_count": 127275,
            "event_true_normal_count": 2433094
          },
          "reason_counts": {
            "closest_cluster_outside_epsilon": 285628,
            "known_benign_cluster": 1562945,
            "known_malicious_cluster": 174780,
            "not_confident_enough": 537016
          },
          "sequence_abstained_count": 34772,
          "sequence_confident_anomaly_count": 20630,
          "sequence_confident_normal_count": 0
        },
        "source": "prediction_diagnostics.event_decision_metrics"
      },
      "evaluation_unit_count": 2560369,
      "headline_metrics": {
        "accuracy": 0.93754363,
        "f1": 0.55270728,
        "precision": 0.38365374,
        "recall": 0.98810822
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "diagnostic_only"
    },
    "manual_workload_reduction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "alert_count": 7690,
        "alerts_per_cluster": 10,
        "cluster_count": 769,
        "coverage": 0.9177141645113307,
        "covered_contextual_sequence_count": 2007586,
        "mode": "manual",
        "overall": 0.9141988869963988,
        "reduction": 0.9961695289765918,
        "total_contextual_sequence_count": 2187594,
        "uncovered_contextual_sequence_count": 180008
      },
      "evaluation_unit_count": null,
      "headline_metrics": {},
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "cluster",
      "metric_scope": "manual_workload_reduction",
      "prediction_unit": "cluster",
      "status": "valid"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.6563143828096654,
          "f1": 0.2525511819151569,
          "precision": 0.2834049622169783,
          "recall": 0.2458591489409943
        },
        "classification_top1_weighted": {
          "accuracy": 0.6563143828096654,
          "f1": 0.6245893720140351,
          "precision": 0.7027837790480659,
          "recall": 0.6563143828096654
        },
        "exclusions": {
          "insufficient_history": 0,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": null,
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.6563143828096654,
            "2": 0.7663036070191445,
            "3": 0.836971545898267,
            "5": 0.883293384664476
          },
          "hit_count": {
            "1": 1680407,
            "2": 1962020,
            "3": 2142956,
            "5": 2261557
          },
          "k_values": [
            1,
            2,
            3,
            5
          ]
        },
        "totals": {
          "coverage": 1.0,
          "events_eligible": 2560369,
          "events_seen": 2560369
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 1.0
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "semi_automatic_workload_reduction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "alert_count": null,
        "alerts_per_cluster": null,
        "cluster_count": 769,
        "coverage": 0.6787009997387096,
        "covered_contextual_sequence_count": 1737725,
        "mode": "semi_automatic",
        "overall": 0.6787009997387096,
        "reduction": 1.0,
        "total_contextual_sequence_count": 2560369,
        "uncovered_contextual_sequence_count": 822644
      },
      "evaluation_unit_count": null,
      "headline_metrics": {},
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "cluster",
      "metric_scope": "semi_automatic_workload_reduction",
      "prediction_unit": "cluster",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 34772,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 25755,
        "normal": 29647
      },
      "confusion_matrix": {
        "fn": 0,
        "fp": 789,
        "tn": 0,
        "tp": 19841
      },
      "counted_predictions": 20630,
      "diagnostics": {
        "abstain_rate": 0.6276307714522942,
        "auto_coverage": 0.3723692285477059,
        "class_counts": {
          "anomalous": 25755,
          "normal": 29647
        }
      },
      "evaluation_unit_count": 55402,
      "headline_metrics": {
        "accuracy": 0.96175473,
        "f1": 0.98050456,
        "precision": 0.96175473,
        "recall": 1.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "valid"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 69252,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 29647,
    "1": 25755
  },
  "test_sequence_count": 55402,
  "train_label_counts": {
    "0": 8230,
    "1": 5620
  },
  "train_sequence_count": 13850
}
```


---

## `bgl_entity_chronological_markov/afa1f3a04d1c/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "0c5fe4c35909cbd2671fdb4f4cc52e736410e58c165b453c7d8eaa9d4f3825c9",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 29647,
      "1": 25755
    },
    "train": {
      "0": 8230,
      "1": 5620
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "markov",
    "ignored_sequence_count": 0,
    "normal_context_vocabulary": 33,
    "normal_sequence_count": 8230,
    "normal_template_vocabulary": 33,
    "normal_transition_count": 847825,
    "order": 1,
    "score_threshold": 0.7441658918041993,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 29647,
      "1": 25755
    },
    "test_sequence_count": 55402,
    "threshold_source": "train_score_quantile",
    "train_label_counts": {
      "0": 8230,
      "1": 5620
    },
    "train_sequence_count": 13850
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "afa1f3a04d1c1cfe46ba507dc5ea03ea368a7e8b474a05685e2d43c616dd78b0",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 69252,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 55402,
    "train": 13850
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999422,
    "eligible_train_sequence_count": 13850,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 13850,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 13850
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_entity_chronological_markov/afa1f3a04d1c/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 1.51327508,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 25755,
        "normal": 29647
      },
      "confusion_matrix": {
        "fn": 384,
        "fp": 28581,
        "tn": 1066,
        "tp": 25371
      },
      "counted_predictions": 55402,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 25755,
          "normal": 29647
        }
      },
      "evaluation_unit_count": 55402,
      "headline_metrics": {
        "accuracy": 0.47718494,
        "f1": 0.63660657,
        "precision": 0.47025133,
        "recall": 0.98509027
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 69252,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 29647,
    "1": 25755
  },
  "test_sequence_count": 55402,
  "train_label_counts": {
    "0": 8230,
    "1": 5620
  },
  "train_sequence_count": 13850
}
```


---

## `bgl_entity_chronological_naive_bayes/8cb80cdd5ac5/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "0c5fe4c35909cbd2671fdb4f4cc52e736410e58c165b453c7d8eaa9d4f3825c9",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 29647,
      "1": 25755
    },
    "train": {
      "0": 8230,
      "1": 5620
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "anomalous_posterior_threshold": 0.5,
    "class_priors": {
      "0": 0.5942238267148015,
      "1": 0.40577617328519855
    },
    "detector": "naive_bayes",
    "ignored_sequence_count": 0,
    "key_phrases_by_class": {
      "anomalous": [
        "ras app fatal ciod: error reading message prefix after <:*:> message on ciostream socket to <:ip:>:<:num:>: <:*:> <:*:> <:*:> <:*:>",
        "ras mmcs error idoproxydb hit assert condition: assert <:*:> source <:*:> source line=<:num:> function=int <:*:> <:*:>",
        "ras <:*:> <:*:> ciod: <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> <:*:> no such file or directory",
        "ras app fatal ciod: error loading /home/greeno/ dev2/ pasci2/ hyperclaw.mf.rich/run/ jag work/raptor.dev.new3d.bgl.cxx xl.mpi.ex: invalid or missing program image, no such file or directory",
        "ras app fatal ciod: error reading message prefix on ciostream socket to <:ip:>:<:num:>, <:*:> <:*:> <:*:> <:*:>"
      ],
      "normal": [
        "ras kernel info kernel detected <:num:> integer alignment exceptions (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:> (<:num:>) iar <:hex:>, dear <:hex:>",
        "ras kernel info <:num:> microseconds spent in the rbs signal handler during <:num:> calls. <:num:> microseconds was the maximum time for a single instance of a correctable ddr.",
        "ras kernel info <:num:> total interrupts. <:num:> critical input interrupts. <:num:> microseconds total spent on critical input interrupts, <:num:> microseconds max time in a critical input interrupt.",
        "ras kernel info <:num:> <:*:> <:*:> error(s) (dcr <:hex:>) detected and corrected over <:num:> seconds",
        "ras kernel info l1 dcache summary averages: #ofdirtylines: <:num:> out of <:num:> #ofdirtydblword: <:num:> out of <:num:>"
      ]
    },
    "phrase_ngram_max": 2,
    "phrase_ngram_min": 1,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 29647,
      "1": 25755
    },
    "test_sequence_count": 55402,
    "top_k_phrases": 5,
    "train_label_counts": {
      "0": 8230,
      "1": 5620
    },
    "train_sequence_count": 13850,
    "train_template_phrase_vocabulary": 1501
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "8cb80cdd5ac5620e918b43130293c9460c11c7d9074ff27c4272179c2ed17a76",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 69252,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 55402,
    "train": 13850
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999422,
    "eligible_train_sequence_count": 13850,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 13850,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 13850
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_entity_chronological_naive_bayes/8cb80cdd5ac5/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.13654273,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 25755,
        "normal": 29647
      },
      "confusion_matrix": {
        "fn": 20237,
        "fp": 2028,
        "tn": 27619,
        "tp": 5518
      },
      "counted_predictions": 55402,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 25755,
          "normal": 29647
        }
      },
      "evaluation_unit_count": 55402,
      "headline_metrics": {
        "accuracy": 0.5981192,
        "f1": 0.33140146,
        "precision": 0.73124834,
        "recall": 0.21424966
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 69252,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 29647,
    "1": 25755
  },
  "test_sequence_count": 55402,
  "train_label_counts": {
    "0": 8230,
    "1": 5620
  },
  "train_sequence_count": 13850
}
```


---

## `bgl_entity_chronological_template_frequency/43f5f1e99735/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/bgl_entity",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity"
  },
  "dataset_fingerprint": "0c5fe4c35909cbd2671fdb4f4cc52e736410e58c165b453c7d8eaa9d4f3825c9",
  "dataset_name": "BGL",
  "dataset_variant": "bgl_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 29647,
      "1": 25755
    },
    "train": {
      "0": 8230,
      "1": 5620
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "template_frequency",
    "ignored_sequence_count": 0,
    "score_threshold": 1.7770339559626627,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 29647,
      "1": 25755
    },
    "test_sequence_count": 55402,
    "threshold_source": "train_score_quantile",
    "train_event_count": 2187594,
    "train_label_counts": {
      "0": 8230,
      "1": 5620
    },
    "train_sequence_count": 13850,
    "train_template_vocabulary": 168
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/bgl_entity/BGL/BGL.log",
    "sha256": "666130b15ef44eb32fd02bd053e6c6e007c37696b5e7e8b9d8e45b729876a5d2"
  },
  "run_fingerprint": "43f5f1e99735b1b2900cbc9409e6cb384b8a0f9b6e9b7883eca02e5865361371",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 69252,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 55402,
    "train": 13850
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999422,
    "eligible_train_sequence_count": 13850,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 13850,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 13850
  },
  "source": {
    "preset": "bgl",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "bgl",
  "structured_rows": 4747963,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1136361605233,
    "min_unix_ms": 1117813370363
  }
}
```


---

## `bgl_entity_chronological_template_frequency/43f5f1e99735/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 3.63800319,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 25755,
        "normal": 29647
      },
      "confusion_matrix": {
        "fn": 2826,
        "fp": 28865,
        "tn": 782,
        "tp": 22929
      },
      "counted_predictions": 55402,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 25755,
          "normal": 29647
        }
      },
      "evaluation_unit_count": 55402,
      "headline_metrics": {
        "accuracy": 0.42798094,
        "f1": 0.59134225,
        "precision": 0.44269607,
        "recall": 0.89027373
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 69252,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 29647,
    "1": 25755
  },
  "test_sequence_count": 55402,
  "train_label_counts": {
    "0": 8230,
    "1": 5620
  },
  "train_sequence_count": 13850
}
```


---

## `hdfs_v1_entity_chronological_markov/5d815e33b8a9/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/data/hs1822/.cache/hdfs_entity",
    "data_root": "/data/hs1822/hdfs_entity"
  },
  "dataset_fingerprint": "3f3035372508752047473fc8d4ce40f875d57238974fca924642e070059c3ddc",
  "dataset_name": "HDFS_V1",
  "dataset_variant": "hdfs_v1_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 448975,
      "1": 11074
    },
    "train": {
      "0": 109248,
      "1": 5764
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "markov",
    "ignored_sequence_count": 0,
    "normal_context_vocabulary": 20,
    "normal_sequence_count": 109248,
    "normal_template_vocabulary": 20,
    "normal_transition_count": 2529775,
    "order": 1,
    "score_threshold": 1.127598551055294,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 448975,
      "1": 11074
    },
    "test_sequence_count": 460049,
    "threshold_source": "train_score_quantile",
    "train_label_counts": {
      "0": 109248,
      "1": 5764
    },
    "train_sequence_count": 115012
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/data/hs1822/hdfs_entity/HDFS_V1/HDFS.log",
    "sha256": "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
  },
  "run_fingerprint": "5d815e33b8a9460c744c653f94a85cb3103ef8b7e024b8f55a897ad89fed0830",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575061,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 460049,
    "train": 115012
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999965,
    "eligible_train_sequence_count": 115012,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 115012,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 115012
  },
  "source": {
    "preset": "hdfs_v1",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_v1",
  "structured_rows": 11175629,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1226402188000,
    "min_unix_ms": 1226262918000
  }
}
```


---

## `hdfs_v1_entity_chronological_markov/5d815e33b8a9/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.98094495,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 11074,
        "normal": 448975
      },
      "confusion_matrix": {
        "fn": 335,
        "fp": 22486,
        "tn": 426489,
        "tp": 10739
      },
      "counted_predictions": 460049,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 11074,
          "normal": 448975
        }
      },
      "evaluation_unit_count": 460049,
      "headline_metrics": {
        "accuracy": 0.95039441,
        "f1": 0.48484164,
        "precision": 0.32322047,
        "recall": 0.96974896
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 575061,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 448975,
    "1": 11074
  },
  "test_sequence_count": 460049,
  "train_label_counts": {
    "0": 109248,
    "1": 5764
  },
  "train_sequence_count": 115012
}
```


---

## `hdfs_v1_entity_chronological_naive_bayes/a399a01338e9/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/data/hs1822/.cache/hdfs_entity",
    "data_root": "/data/hs1822/hdfs_entity"
  },
  "dataset_fingerprint": "3f3035372508752047473fc8d4ce40f875d57238974fca924642e070059c3ddc",
  "dataset_name": "HDFS_V1",
  "dataset_variant": "hdfs_v1_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 448975,
      "1": 11074
    },
    "train": {
      "0": 109248,
      "1": 5764
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "anomalous_posterior_threshold": 0.5,
    "class_priors": {
      "0": 0.9498834904183912,
      "1": 0.050116509581608876
    },
    "detector": "naive_bayes",
    "ignored_sequence_count": 0,
    "key_phrases_by_class": {
      "anomalous": [
        "info dfs.fsnamesystem: block* namesystem.addstoredblock: addstoredblock request received for blk <:*:> on <:ip:>:<:num:> size <:num:> but it does not belong to any file.",
        "info dfs.datanode$dataxceiver: writeblock blk <:*:> received exception java.io.ioexception: could not read from stream",
        "warn dfs.fsnamesystem: block* namesystem.addstoredblock: redundant addstoredblock request received for blk <:*:> on <:ip:>:<:num:> size <:num:>",
        "info <:*:> <:*:> blk <:*:> <:*:> <:*:> java.net.sockettimeoutexception: <:num:> millis timeout while waiting for channel to be ready for <:*:> ch : java.nio.channels.socketchannel[connected local=/<:ip:>:<:num:> remote=/<:ip:>:<:num:>]",
        "info <:*:> <:*:> blk <:*:> <:*:> <:*:> java.io.interruptedioexception: interruped while waiting for io on channel java.nio.channels.socketchannel[connected local=/<:ip:>:<:num:> remote=/<:ip:>:<:num:>]. <:num:> millis timeout left."
      ],
      "normal": [
        "warn dfs.datanode$dataxceiver: <:ip:>:<:num:>:got exception while serving blk <:*:> to /<:ip:>:",
        "info dfs.datanode$dataxceiver: <:ip:>:<:num:> served block blk <:*:> to /<:ip:>",
        "info dfs.fsnamesystem: block* namesystem.delete: blk <:*:> is added to invalidset of <:ip:>:<:num:>",
        "info dfs.datanode$packetresponder: received block blk <:*:> of size <:num:> from /<:ip:>",
        "info dfs.datanode$packetresponder: packetresponder <:num:> for block blk <:*:> <:*:>"
      ]
    },
    "phrase_ngram_max": 2,
    "phrase_ngram_min": 1,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 448975,
      "1": 11074
    },
    "test_sequence_count": 460049,
    "top_k_phrases": 5,
    "train_label_counts": {
      "0": 109248,
      "1": 5764
    },
    "train_sequence_count": 115012,
    "train_template_phrase_vocabulary": 405
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/data/hs1822/hdfs_entity/HDFS_V1/HDFS.log",
    "sha256": "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
  },
  "run_fingerprint": "a399a01338e9f2f5fdfa2220c15885f67f9b29fe742f934d22c967e20c477ca4",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575061,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 460049,
    "train": 115012
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999965,
    "eligible_train_sequence_count": 115012,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 115012,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 115012
  },
  "source": {
    "preset": "hdfs_v1",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_v1",
  "structured_rows": 11175629,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1226402188000,
    "min_unix_ms": 1226262918000
  }
}
```


---

## `hdfs_v1_entity_chronological_naive_bayes/a399a01338e9/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.02659607,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 11074,
        "normal": 448975
      },
      "confusion_matrix": {
        "fn": 1,
        "fp": 1695,
        "tn": 447280,
        "tp": 11073
      },
      "counted_predictions": 460049,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 11074,
          "normal": 448975
        }
      },
      "evaluation_unit_count": 460049,
      "headline_metrics": {
        "accuracy": 0.99631344,
        "f1": 0.92886503,
        "precision": 0.86724624,
        "recall": 0.9999097
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 575061,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 448975,
    "1": 11074
  },
  "test_sequence_count": 460049,
  "train_label_counts": {
    "0": 109248,
    "1": 5764
  },
  "train_sequence_count": 115012
}
```


---

## `hdfs_v1_entity_chronological_template_frequency/f8a73b0a8c48/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/data/hs1822/.cache/hdfs_entity",
    "data_root": "/data/hs1822/hdfs_entity"
  },
  "dataset_fingerprint": "3f3035372508752047473fc8d4ce40f875d57238974fca924642e070059c3ddc",
  "dataset_name": "HDFS_V1",
  "dataset_variant": "hdfs_v1_entity_chronological",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 448975,
      "1": 11074
    },
    "train": {
      "0": 109248,
      "1": 5764
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "template_frequency",
    "ignored_sequence_count": 0,
    "score_threshold": 2.3558842630402745,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 448975,
      "1": 11074
    },
    "test_sequence_count": 460049,
    "threshold_source": "train_score_quantile",
    "train_event_count": 2730412,
    "train_label_counts": {
      "0": 109248,
      "1": 5764
    },
    "train_sequence_count": 115012,
    "train_template_vocabulary": 40
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/data/hs1822/hdfs_entity/HDFS_V1/HDFS.log",
    "sha256": "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52"
  },
  "run_fingerprint": "f8a73b0a8c4887d3d862326d6f7ef6da23457e7fbbe33b660741256681ef073b",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575061,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 460049,
    "train": 115012
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.19999965,
    "eligible_train_sequence_count": 115012,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 115012,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 115012
  },
  "source": {
    "preset": "hdfs_v1",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_v1",
  "structured_rows": 11175629,
  "template_parser": "drain3",
  "timestamp_bounds": {
    "max_unix_ms": 1226402188000,
    "min_unix_ms": 1226262918000
  }
}
```


---

## `hdfs_v1_entity_chronological_template_frequency/f8a73b0a8c48/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 2.18723437,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 11074,
        "normal": 448975
      },
      "confusion_matrix": {
        "fn": 0,
        "fp": 2184,
        "tn": 446791,
        "tp": 11074
      },
      "counted_predictions": 460049,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 11074,
          "normal": 448975
        }
      },
      "evaluation_unit_count": 460049,
      "headline_metrics": {
        "accuracy": 0.99525268,
        "f1": 0.91024166,
        "precision": 0.83526927,
        "recall": 1.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 575061,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 448975,
    "1": 11074
  },
  "test_sequence_count": 460049,
  "train_label_counts": {
    "0": 109248,
    "1": 5764
  },
  "train_sequence_count": 115012
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_deeplog_parameter_detection_enabled_false/e5535d84db4a/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/hdfs_wuyifan18_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed"
  },
  "dataset_fingerprint": "eb56e0a7e2c77e92e6c56f13b64e4cd6e0024aed0c8a27f7b5a97c4e8705b1ba",
  "dataset_name": "HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED",
  "dataset_variant": "hdfs_wuyifan18_preprocessed_exact_boundary",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 553366,
      "1": 16838
    },
    "train": {
      "0": 4855
    }
  },
  "label_unit": "next_event",
  "model_manifest": {
    "batch_size": 2048,
    "detector": "deeplog",
    "epochs": 300,
    "gaussian_confidence": 0.99,
    "hidden_size": 64,
    "history_size": 10,
    "ignored_sequence_count": 0,
    "implementation_scope": "Scoped DeepLog core v1",
    "include_elapsed_time": true,
    "learning_rate": 0.001,
    "num_layers": 2,
    "parameter_detection_enabled": false,
    "parameter_models": [],
    "parameter_schema_policy": "disabled for this reproduction",
    "parameter_validation_policy": "not applicable: HDFS paper reproduction uses key-only anomaly detection",
    "scored_parameter_event_count": 0,
    "scored_parameter_event_fraction": 0.0,
    "skipped_parameter_model_count": 0,
    "skipped_parameter_models": [],
    "test_label_counts": {
      "0": 553366,
      "1": 16838
    },
    "test_sequence_count": 570204,
    "top_g": 11,
    "top_g_values": [
      1,
      3,
      5,
      7,
      9,
      11
    ],
    "train_key_vocabulary_size": 14,
    "train_label_counts": {
      "0": 4855
    },
    "train_parameter_covered_event_count": 0,
    "train_parameter_covered_event_fraction": 0.0,
    "train_sequence_count": 4855,
    "trained_parameter_model_count": 0,
    "validation_fraction": 0.1
  },
  "prediction_unit": "next_event",
  "primary_metric_scope": "next_event_prediction",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed/HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED/preprocessed/hdfs_events.log",
    "sha256": "db2abb093d40b0111b021659b399f05770ac295b4aaf7bd293cc5d691eef17c7"
  },
  "run_fingerprint": "e5535d84db4ac7d19f19de551b3e5c0a0715a2e5dca923c38424b3451a805f08",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575059,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 570204,
    "train": 4855
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.00844261,
    "eligible_train_sequence_count": 4855,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 4855,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 4855
  },
  "source": {
    "preset": "hdfs_wuyifan18_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_wuyifan18_deeplog_preprocessed",
  "structured_rows": 11172157,
  "template_parser": "identity",
  "timestamp_bounds": {
    "max_unix_ms": null,
    "min_unix_ms": null
  }
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_deeplog_parameter_detection_enabled_false/e5535d84db4a/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "next_event",
  "mean_test_score": 0.01876122,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 162697,
        "normal": 5258554
      },
      "confusion_matrix": {
        "fn": 130276,
        "fp": 860,
        "tn": 5257694,
        "tp": 32421
      },
      "counted_predictions": 5421251,
      "diagnostics": {
        "events_eligible": 5421251,
        "events_seen": 5421251,
        "source": "event_level_detection"
      },
      "evaluation_unit_count": 5421251,
      "headline_metrics": {
        "accuracy": 0.97581075,
        "f1": 0.33086367,
        "precision": 0.97415943,
        "recall": 0.19927227
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "diagnostic_only"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.9086472845474227,
          "f1": 0.3229075715049904,
          "precision": 0.33599949350353525,
          "recall": 0.32557563144048934
        },
        "classification_top1_weighted": {
          "accuracy": 0.9086472845474227,
          "f1": 0.9039597475421964,
          "precision": 0.9059008052203318,
          "recall": 0.9086472845474227
        },
        "exclusions": {
          "insufficient_history": 5655781,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": {
          "expected_insufficient_history_from_segments": 5655781,
          "history_size": 10,
          "largest_segments": [
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 298,
              "segment_id": 290607,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 284,
              "segment_id": 569510,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 280,
              "segment_id": 488305,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 277,
              "segment_id": 556614,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 277,
              "segment_id": 201946,
              "source_object_type": "entity"
            }
          ],
          "segment_count": 570204,
          "segment_length_histogram": {
            "10": 2,
            "11": 8,
            "13": 95485,
            "14": 5542,
            "15": 135,
            "16": 85,
            "17": 12,
            "19": 297918,
            "2": 2952,
            "20": 24893,
            "21": 4377,
            "22": 26426,
            "222": 16,
            "223": 1,
            "229": 3,
            "23": 21182,
            "230": 1,
            "24": 11673,
            "25": 29614,
            "26": 7065,
            "269": 14,
            "27": 2675,
            "270": 1,
            "273": 2,
            "274": 1,
            "277": 2,
            "28": 13125,
            "280": 1,
            "284": 1,
            "29": 1978,
            "298": 1,
            "3": 3229,
            "30": 1462,
            "31": 12938,
            "32": 1685,
            "33": 334,
            "34": 94,
            "35": 1471,
            "36": 1326,
            "37": 982,
            "38": 663,
            "39": 103,
            "40": 62,
            "41": 370,
            "42": 180,
            "43": 43,
            "44": 24,
            "45": 9,
            "46": 9,
            "47": 1,
            "48": 4,
            "49": 2,
            "50": 1,
            "51": 1,
            "52": 1,
            "53": 2,
            "54": 2,
            "55": 1,
            "56": 1,
            "58": 1,
            "6": 10,
            "61": 2
          },
          "smallest_segments": [
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 553369,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 553372,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 553373,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 553375,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 553392,
              "source_object_type": "entity"
            }
          ]
        },
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.9086472845474227,
            "11": 0.9938610110470812,
            "3": 0.9802896047425216,
            "5": 0.9916878963914417,
            "7": 0.9924224132031518,
            "9": 0.9930840686033537
          },
          "hit_count": {
            "1": 4926005,
            "11": 5387970,
            "3": 5314396,
            "5": 5376189,
            "7": 5380171,
            "9": 5383758
          },
          "k_values": [
            1,
            3,
            5,
            7,
            9,
            11
          ]
        },
        "totals": {
          "coverage": 0.4894136804876974,
          "events_eligible": 5421251,
          "events_seen": 11077032
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 0.4894136804876974
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 16838,
        "normal": 553366
      },
      "confusion_matrix": {
        "fn": 6546,
        "fp": 406,
        "tn": 552960,
        "tp": 10292
      },
      "counted_predictions": 570204,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 16838,
          "normal": 553366
        }
      },
      "evaluation_unit_count": 570204,
      "headline_metrics": {
        "accuracy": 0.98780787,
        "f1": 0.74753051,
        "precision": 0.96204898,
        "recall": 0.61123649
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "next_event",
  "primary_metric_scope": "next_event_prediction",
  "sequence_count": 575059,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 553366,
    "1": 16838
  },
  "test_sequence_count": 570204,
  "train_label_counts": {
    "0": 4855
  },
  "train_sequence_count": 4855
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_markov/4f2e65c741b7/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/hdfs_wuyifan18_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed"
  },
  "dataset_fingerprint": "eb56e0a7e2c77e92e6c56f13b64e4cd6e0024aed0c8a27f7b5a97c4e8705b1ba",
  "dataset_name": "HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED",
  "dataset_variant": "hdfs_wuyifan18_preprocessed_exact_boundary",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 553366,
      "1": 16838
    },
    "train": {
      "0": 4855
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "markov",
    "ignored_sequence_count": 0,
    "normal_context_vocabulary": 14,
    "normal_sequence_count": 4855,
    "normal_template_vocabulary": 14,
    "normal_transition_count": 90270,
    "order": 1,
    "score_threshold": 1.124161354746077,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 553366,
      "1": 16838
    },
    "test_sequence_count": 570204,
    "threshold_source": "train_score_quantile",
    "train_label_counts": {
      "0": 4855
    },
    "train_sequence_count": 4855
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed/HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED/preprocessed/hdfs_events.log",
    "sha256": "db2abb093d40b0111b021659b399f05770ac295b4aaf7bd293cc5d691eef17c7"
  },
  "run_fingerprint": "4f2e65c741b7776032cd845aa71585162d4defdb6b5144b13d7c5ba0ef0a885e",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575059,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 570204,
    "train": 4855
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.00844261,
    "eligible_train_sequence_count": 4855,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 4855,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 4855
  },
  "source": {
    "preset": "hdfs_wuyifan18_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_wuyifan18_deeplog_preprocessed",
  "structured_rows": 11172157,
  "template_parser": "identity",
  "timestamp_bounds": {
    "max_unix_ms": null,
    "min_unix_ms": null
  }
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_markov/4f2e65c741b7/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.75167354,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 16838,
        "normal": 553366
      },
      "confusion_matrix": {
        "fn": 5614,
        "fp": 24082,
        "tn": 529284,
        "tp": 11224
      },
      "counted_predictions": 570204,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 16838,
          "normal": 553366
        }
      },
      "evaluation_unit_count": 570204,
      "headline_metrics": {
        "accuracy": 0.94792039,
        "f1": 0.43050015,
        "precision": 0.3179063,
        "recall": 0.66658748
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 575059,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 553366,
    "1": 16838
  },
  "test_sequence_count": 570204,
  "train_label_counts": {
    "0": 4855
  },
  "train_sequence_count": 4855
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_template_frequency/ad5743055392/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/hdfs_wuyifan18_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed"
  },
  "dataset_fingerprint": "eb56e0a7e2c77e92e6c56f13b64e4cd6e0024aed0c8a27f7b5a97c4e8705b1ba",
  "dataset_name": "HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED",
  "dataset_variant": "hdfs_wuyifan18_preprocessed_exact_boundary",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 553366,
      "1": 16838
    },
    "train": {
      "0": 4855
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "template_frequency",
    "ignored_sequence_count": 0,
    "score_threshold": 2.4570141071255507,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 553366,
      "1": 16838
    },
    "test_sequence_count": 570204,
    "threshold_source": "train_score_quantile",
    "train_event_count": 95125,
    "train_label_counts": {
      "0": 4855
    },
    "train_sequence_count": 4855,
    "train_template_vocabulary": 14
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/hdfs_wuyifan18_deeplog_preprocessed/HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED/preprocessed/hdfs_events.log",
    "sha256": "db2abb093d40b0111b021659b399f05770ac295b4aaf7bd293cc5d691eef17c7"
  },
  "run_fingerprint": "ad574305539233a03f2aeae11c75b9d17debf05835be483611f7ceb9c3bc0db6",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 575059,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 570204,
    "train": 4855
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.00844261,
    "eligible_train_sequence_count": 4855,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 4855,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 4855
  },
  "source": {
    "preset": "hdfs_wuyifan18_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "hdfs_wuyifan18_deeplog_preprocessed",
  "structured_rows": 11172157,
  "template_parser": "identity",
  "timestamp_bounds": {
    "max_unix_ms": null,
    "min_unix_ms": null
  }
}
```


---

## `hdfs_wuyifan18_preprocessed_exact_boundary_template_frequency/ad5743055392/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 2.09934733,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 16838,
        "normal": 553366
      },
      "confusion_matrix": {
        "fn": 6448,
        "fp": 22682,
        "tn": 530684,
        "tp": 10390
      },
      "counted_predictions": 570204,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 16838,
          "normal": 553366
        }
      },
      "evaluation_unit_count": 570204,
      "headline_metrics": {
        "accuracy": 0.94891302,
        "f1": 0.41634943,
        "precision": 0.31416304,
        "recall": 0.61705666
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 575059,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 553366,
    "1": 16838
  },
  "test_sequence_count": 570204,
  "train_label_counts": {
    "0": 4855
  },
  "train_sequence_count": 4855
}
```


---

## `openstack_deeplog_preprocessed_deepcase/5f781aad7bf9/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction",
    "manual_workload_reduction",
    "semi_automatic_workload_reduction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/openstack_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed"
  },
  "dataset_fingerprint": "881afa2a6864c5bea814bf8bc6c1c76e1ca1bd096af7ed400991745149e1bfde",
  "dataset_name": "OPENSTACK_DEEPLOG_PREPROCESSED",
  "dataset_variant": "openstack_deeplog_preprocessed",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 1315,
      "1": 198
    },
    "train": {
      "0": 557
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "batch_size": 128,
    "cluster_score_strategy": "max",
    "clustered_sample_count": 10520,
    "confidence_threshold": 0.2,
    "context_length": 10,
    "detector": "deepcase",
    "device": "cuda",
    "epochs": 100,
    "eps": 0.1,
    "hidden_size": 128,
    "ignored_sequence_count": 0,
    "implementation_scope": "Official DeepCase library integration",
    "iterations": 100,
    "known_benign_cluster_count": 10520,
    "known_cluster_count": 25,
    "known_malicious_cluster_count": 0,
    "label_policy": "event-label supervision when available: each event-centered sample uses its target event label and falls back to the parent TemplateSequence label when the event label is missing",
    "label_smoothing_delta": 0.1,
    "learning_rate": 0.01,
    "min_samples": 5,
    "no_score": -1,
    "online_updates_status": "not implemented",
    "persistent_cluster_database_status": "not implemented",
    "prediction_diagnostics": {
      "abstained_anomalous_label_count": 198,
      "abstained_event_count": 34786,
      "abstained_normal_label_count": 1315,
      "confident_anomaly_event_count": 0,
      "confident_event_count": 4411,
      "event_count": 39197,
      "event_decision_metrics": {
        "event_abstain_rate": 0.8874658774906243,
        "event_abstained_decision_count": 34786,
        "event_accuracy": 0.8712310133756518,
        "event_auto_coverage": 0.11253412250937572,
        "event_auto_decision_count": 4411,
        "event_count": 39197,
        "event_f1": 0.0,
        "event_fn": 568,
        "event_fp": 0,
        "event_precision": 0.0,
        "event_predicted_anomalous_count": 0,
        "event_predicted_normal_count": 4411,
        "event_recall": 0.0,
        "event_tn": 3843,
        "event_tp": 0,
        "event_true_anomalous_count": 5100,
        "event_true_normal_count": 34097
      },
      "reason_counts": {
        "closest_cluster_outside_epsilon": 24210,
        "known_benign_cluster": 4411,
        "not_confident_enough": 10576
      },
      "sequence_abstained_count": 1513,
      "sequence_confident_anomaly_count": 0,
      "sequence_confident_normal_count": 0
    },
    "query_batch_size": 1024,
    "random_seed": 0,
    "teach_ratio": 0.5,
    "test_label_counts": {
      "0": 1315,
      "1": 198
    },
    "test_sequence_count": 1513,
    "timeout_seconds": 86400.0,
    "train_event_vocabulary_size": 24,
    "train_label_counts": {
      "0": 557
    },
    "train_sample_count": 14421,
    "train_sequence_count": 557,
    "unknown_cluster_score_count": 3901
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed/OPENSTACK_DEEPLOG_PREPROCESSED/preprocessed/openstack_labelled_raw.log",
    "sha256": "dad8d1091c5ca9054b230847b689d0b1fea78687b903fe032f2e0f087c2f21bf"
  },
  "run_fingerprint": "5f781aad7bf922781fd44ec2c68364e9c88aa4df837403aa077b21452d9affeb",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 2070,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 1513,
    "train": 557
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.26908213,
    "eligible_train_sequence_count": 557,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 557,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 557
  },
  "source": {
    "preset": "openstack_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "openstack_deeplog_preprocessed",
  "structured_rows": 53618,
  "template_parser": "spell",
  "timestamp_bounds": {
    "max_unix_ms": 1495002306749,
    "min_unix_ms": 1494790742007
  }
}
```


---

## `openstack_deeplog_preprocessed_deepcase/5f781aad7bf9/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction",
    "manual_workload_reduction",
    "semi_automatic_workload_reduction"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.0,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 34786,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 5100,
        "normal": 34097
      },
      "confusion_matrix": {
        "fn": 568,
        "fp": 0,
        "tn": 3843,
        "tp": 0
      },
      "counted_predictions": 4411,
      "diagnostics": {
        "event_abstain_rate": 0.8874658774906243,
        "event_auto_coverage": 0.11253412250937572,
        "event_decision_metrics": {
          "event_abstain_rate": 0.8874658774906243,
          "event_abstained_decision_count": 34786,
          "event_accuracy": 0.8712310133756518,
          "event_auto_coverage": 0.11253412250937572,
          "event_auto_decision_count": 4411,
          "event_count": 39197,
          "event_f1": 0.0,
          "event_fn": 568,
          "event_fp": 0,
          "event_precision": 0.0,
          "event_predicted_anomalous_count": 0,
          "event_predicted_normal_count": 4411,
          "event_recall": 0.0,
          "event_tn": 3843,
          "event_tp": 0,
          "event_true_anomalous_count": 5100,
          "event_true_normal_count": 34097
        },
        "events_eligible": 4411,
        "events_seen": 39197,
        "prediction_diagnostics": {
          "abstained_anomalous_label_count": 198,
          "abstained_event_count": 34786,
          "abstained_normal_label_count": 1315,
          "confident_anomaly_event_count": 0,
          "confident_event_count": 4411,
          "event_count": 39197,
          "event_decision_metrics": {
            "event_abstain_rate": 0.8874658774906243,
            "event_abstained_decision_count": 34786,
            "event_accuracy": 0.8712310133756518,
            "event_auto_coverage": 0.11253412250937572,
            "event_auto_decision_count": 4411,
            "event_count": 39197,
            "event_f1": 0.0,
            "event_fn": 568,
            "event_fp": 0,
            "event_precision": 0.0,
            "event_predicted_anomalous_count": 0,
            "event_predicted_normal_count": 4411,
            "event_recall": 0.0,
            "event_tn": 3843,
            "event_tp": 0,
            "event_true_anomalous_count": 5100,
            "event_true_normal_count": 34097
          },
          "reason_counts": {
            "closest_cluster_outside_epsilon": 24210,
            "known_benign_cluster": 4411,
            "not_confident_enough": 10576
          },
          "sequence_abstained_count": 1513,
          "sequence_confident_anomaly_count": 0,
          "sequence_confident_normal_count": 0
        },
        "source": "prediction_diagnostics.event_decision_metrics"
      },
      "evaluation_unit_count": 39197,
      "headline_metrics": {
        "accuracy": 0.87123101,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "diagnostic_only"
    },
    "manual_workload_reduction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "alert_count": 250,
        "alerts_per_cluster": 10,
        "cluster_count": 25,
        "coverage": 0.7294917134734068,
        "covered_contextual_sequence_count": 10520,
        "mode": "manual",
        "overall": 0.7121558837805977,
        "reduction": 0.9762357414448669,
        "total_contextual_sequence_count": 14421,
        "uncovered_contextual_sequence_count": 3901
      },
      "evaluation_unit_count": null,
      "headline_metrics": {},
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "cluster",
      "metric_scope": "manual_workload_reduction",
      "prediction_unit": "cluster",
      "status": "valid"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.6572952011633544,
          "f1": 0.5401223010487525,
          "precision": 0.5372972493965132,
          "recall": 0.5867882287213689
        },
        "classification_top1_weighted": {
          "accuracy": 0.6572952011633544,
          "f1": 0.6185236398012787,
          "precision": 0.628352049706834,
          "recall": 0.6572952011633544
        },
        "exclusions": {
          "insufficient_history": 0,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": null,
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.6572952011633544,
            "2": 0.77809526239253,
            "3": 0.8312115723142077,
            "5": 0.9172130520192872
          },
          "hit_count": {
            "1": 25764,
            "2": 30499,
            "3": 32581,
            "5": 35952
          },
          "k_values": [
            1,
            2,
            3,
            5
          ]
        },
        "totals": {
          "coverage": 1.0,
          "events_eligible": 39197,
          "events_seen": 39197
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 1.0
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "semi_automatic_workload_reduction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "alert_count": null,
        "alerts_per_cluster": null,
        "cluster_count": 25,
        "coverage": 0.11253412250937572,
        "covered_contextual_sequence_count": 4411,
        "mode": "semi_automatic",
        "overall": 0.11253412250937572,
        "reduction": 1.0,
        "total_contextual_sequence_count": 39197,
        "uncovered_contextual_sequence_count": 34786
      },
      "evaluation_unit_count": null,
      "headline_metrics": {},
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "cluster",
      "metric_scope": "semi_automatic_workload_reduction",
      "prediction_unit": "cluster",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 1513,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 198,
        "normal": 1315
      },
      "confusion_matrix": {
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "tp": 0
      },
      "counted_predictions": 0,
      "diagnostics": {
        "abstain_rate": 1.0,
        "auto_coverage": 0.0,
        "class_counts": {
          "anomalous": 198,
          "normal": 1315
        }
      },
      "evaluation_unit_count": 1513,
      "headline_metrics": {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "valid"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 2070,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 1315,
    "1": 198
  },
  "test_sequence_count": 1513,
  "train_label_counts": {
    "0": 557
  },
  "train_sequence_count": 557
}
```


---

## `openstack_deeplog_preprocessed_deeplog_parameter_detection_enabled_false/14fdf9346169/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/openstack_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed"
  },
  "dataset_fingerprint": "881afa2a6864c5bea814bf8bc6c1c76e1ca1bd096af7ed400991745149e1bfde",
  "dataset_name": "OPENSTACK_DEEPLOG_PREPROCESSED",
  "dataset_variant": "openstack_deeplog_preprocessed",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 1315,
      "1": 198
    },
    "train": {
      "0": 557
    }
  },
  "label_unit": "next_event",
  "model_manifest": {
    "batch_size": 2048,
    "detector": "deeplog",
    "epochs": 300,
    "gaussian_confidence": 0.99,
    "hidden_size": 64,
    "history_size": 10,
    "ignored_sequence_count": 0,
    "implementation_scope": "Scoped DeepLog core v1",
    "include_elapsed_time": true,
    "learning_rate": 0.001,
    "num_layers": 2,
    "parameter_detection_enabled": false,
    "parameter_models": [],
    "parameter_schema_policy": "disabled for this reproduction",
    "parameter_validation_policy": "not applicable: HDFS paper reproduction uses key-only anomaly detection",
    "scored_parameter_event_count": 0,
    "scored_parameter_event_fraction": 0.0,
    "skipped_parameter_model_count": 0,
    "skipped_parameter_models": [],
    "test_label_counts": {
      "0": 1315,
      "1": 198
    },
    "test_sequence_count": 1513,
    "top_g": 11,
    "top_g_values": [
      1,
      3,
      5,
      7,
      9,
      11
    ],
    "train_key_vocabulary_size": 24,
    "train_label_counts": {
      "0": 557
    },
    "train_parameter_covered_event_count": 0,
    "train_parameter_covered_event_fraction": 0.0,
    "train_sequence_count": 557,
    "trained_parameter_model_count": 0,
    "validation_fraction": 0.1
  },
  "prediction_unit": "next_event",
  "primary_metric_scope": "next_event_prediction",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed/OPENSTACK_DEEPLOG_PREPROCESSED/preprocessed/openstack_labelled_raw.log",
    "sha256": "dad8d1091c5ca9054b230847b689d0b1fea78687b903fe032f2e0f087c2f21bf"
  },
  "run_fingerprint": "14fdf9346169215631b4ed941565467470ca27202cf97e14a20041c720061b88",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 2070,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 1513,
    "train": 557
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.26908213,
    "eligible_train_sequence_count": 557,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 557,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 557
  },
  "source": {
    "preset": "openstack_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "openstack_deeplog_preprocessed",
  "structured_rows": 53618,
  "template_parser": "spell",
  "timestamp_bounds": {
    "max_unix_ms": 1495002306749,
    "min_unix_ms": 1494790742007
  }
}
```


---

## `openstack_deeplog_preprocessed_deeplog_parameter_detection_enabled_false/14fdf9346169/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection",
    "event_level_detection",
    "next_event_prediction"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "next_event",
  "mean_test_score": 0.00066094,
  "metric_blocks": {
    "event_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 3129,
        "normal": 20964
      },
      "confusion_matrix": {
        "fn": 3129,
        "fp": 8,
        "tn": 20956,
        "tp": 0
      },
      "counted_predictions": 24093,
      "diagnostics": {
        "events_eligible": 24093,
        "events_seen": 24093,
        "source": "event_level_detection"
      },
      "evaluation_unit_count": 24093,
      "headline_metrics": {
        "accuracy": 0.86979621,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "event",
      "metric_scope": "event_level_detection",
      "prediction_unit": "event",
      "status": "diagnostic_only"
    },
    "next_event_prediction": {
      "abstained_prediction_count": null,
      "aggregation_policy": null,
      "class_counts": null,
      "confusion_matrix": null,
      "counted_predictions": null,
      "diagnostics": {
        "classification_top1_macro": {
          "accuracy": 0.9837712198563898,
          "f1": 0.8547284352299908,
          "precision": 0.8553623709474897,
          "recall": 0.8542615060912445
        },
        "classification_top1_weighted": {
          "accuracy": 0.9837712198563898,
          "f1": 0.9837838154982357,
          "precision": 0.9839602095446159,
          "recall": 0.9837712198563898
        },
        "exclusions": {
          "insufficient_history": 15104,
          "unknown_history": 0,
          "unknown_target": 0
        },
        "segment_diagnostics": {
          "expected_insufficient_history_from_segments": 15104,
          "history_size": 10,
          "largest_segments": [
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 28,
              "segment_id": 1253,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 28,
              "segment_id": 944,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 27,
              "segment_id": 1475,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 27,
              "segment_id": 1387,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 27,
              "segment_id": 1372,
              "source_object_type": "entity"
            }
          ],
          "segment_count": 1513,
          "segment_length_histogram": {
            "1": 2,
            "17": 1,
            "19": 2,
            "2": 1,
            "25": 127,
            "26": 1299,
            "27": 79,
            "28": 2
          },
          "smallest_segments": [
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 1,
              "length": 1,
              "segment_id": 2,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 1,
              "length": 1,
              "segment_id": 201,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 2,
              "length": 2,
              "segment_id": 199,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 17,
              "segment_id": 198,
              "source_object_type": "entity"
            },
            {
              "boundary_reason": "standalone_sequence",
              "insufficient_history": 10,
              "length": 19,
              "segment_id": 1,
              "source_object_type": "entity"
            }
          ]
        },
        "task": "next_event_prediction",
        "top_k": {
          "accuracy": {
            "1": 0.9837712198563898,
            "11": 0.9996679533474453,
            "3": 0.9994604241895987,
            "5": 0.9996264475158759,
            "7": 0.9996679533474453,
            "9": 0.9996679533474453
          },
          "hit_count": {
            "1": 23702,
            "11": 24085,
            "3": 24080,
            "5": 24084,
            "7": 24085,
            "9": 24085
          },
          "k_values": [
            1,
            3,
            5,
            7,
            9,
            11
          ]
        },
        "totals": {
          "coverage": 0.6146643875806822,
          "events_eligible": 24093,
          "events_seen": 39197
        },
        "vocabulary_policy": "full_dataset"
      },
      "evaluation_unit_count": null,
      "headline_metrics": {
        "coverage": 0.6146643875806822
      },
      "ignored_prediction_count": null,
      "invalid_reason": null,
      "label_unit": "next_event",
      "metric_scope": "next_event_prediction",
      "prediction_unit": "next_event",
      "status": "valid"
    },
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 198,
        "normal": 1315
      },
      "confusion_matrix": {
        "fn": 198,
        "fp": 1,
        "tn": 1314,
        "tp": 0
      },
      "counted_predictions": 1513,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 198,
          "normal": 1315
        }
      },
      "evaluation_unit_count": 1513,
      "headline_metrics": {
        "accuracy": 0.86847323,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "next_event",
  "primary_metric_scope": "next_event_prediction",
  "sequence_count": 2070,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 1315,
    "1": 198
  },
  "test_sequence_count": 1513,
  "train_label_counts": {
    "0": 557
  },
  "train_sequence_count": 557
}
```


---

## `openstack_deeplog_preprocessed_markov/4b4578011634/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/openstack_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed"
  },
  "dataset_fingerprint": "881afa2a6864c5bea814bf8bc6c1c76e1ca1bd096af7ed400991745149e1bfde",
  "dataset_name": "OPENSTACK_DEEPLOG_PREPROCESSED",
  "dataset_variant": "openstack_deeplog_preprocessed",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 1315,
      "1": 198
    },
    "train": {
      "0": 557
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "markov",
    "ignored_sequence_count": 0,
    "normal_context_vocabulary": 23,
    "normal_sequence_count": 557,
    "normal_template_vocabulary": 24,
    "normal_transition_count": 13864,
    "order": 1,
    "score_threshold": 0.509241748758778,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 1315,
      "1": 198
    },
    "test_sequence_count": 1513,
    "threshold_source": "train_score_quantile",
    "train_label_counts": {
      "0": 557
    },
    "train_sequence_count": 557
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed/OPENSTACK_DEEPLOG_PREPROCESSED/preprocessed/openstack_labelled_raw.log",
    "sha256": "dad8d1091c5ca9054b230847b689d0b1fea78687b903fe032f2e0f087c2f21bf"
  },
  "run_fingerprint": "4b45780116344eaae115172c9e3593698d001e2932d13ce778ff9bfee09b1b3e",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 2070,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 1513,
    "train": 557
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.26908213,
    "eligible_train_sequence_count": 557,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 557,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 557
  },
  "source": {
    "preset": "openstack_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "openstack_deeplog_preprocessed",
  "structured_rows": 53618,
  "template_parser": "spell",
  "timestamp_bounds": {
    "max_unix_ms": 1495002306749,
    "min_unix_ms": 1494790742007
  }
}
```


---

## `openstack_deeplog_preprocessed_markov/4b4578011634/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 0.25729026,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 198,
        "normal": 1315
      },
      "confusion_matrix": {
        "fn": 187,
        "fp": 61,
        "tn": 1254,
        "tp": 11
      },
      "counted_predictions": 1513,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 198,
          "normal": 1315
        }
      },
      "evaluation_unit_count": 1513,
      "headline_metrics": {
        "accuracy": 0.83608724,
        "f1": 0.08148148,
        "precision": 0.15277778,
        "recall": 0.05555556
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 2070,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 1315,
    "1": 198
  },
  "test_sequence_count": 1513,
  "train_label_counts": {
    "0": 557
  },
  "train_sequence_count": 557
}
```


---

## `openstack_deeplog_preprocessed_template_frequency/690e2c0c7a17/dataset_manifest.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "cache_paths": {
    "cache_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/.cache/openstack_deeplog_preprocessed",
    "data_root": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed"
  },
  "dataset_fingerprint": "881afa2a6864c5bea814bf8bc6c1c76e1ca1bd096af7ed400991745149e1bfde",
  "dataset_name": "OPENSTACK_DEEPLOG_PREPROCESSED",
  "dataset_variant": "openstack_deeplog_preprocessed",
  "evaluation_unit": "sequence",
  "label_counts": {
    "test": {
      "0": 1315,
      "1": 198
    },
    "train": {
      "0": 557
    }
  },
  "label_unit": "sequence",
  "model_manifest": {
    "calibration_quantile": 0.95,
    "detector": "template_frequency",
    "ignored_sequence_count": 0,
    "score_threshold": 3.118440895125551,
    "smoothing": 1.0,
    "test_label_counts": {
      "0": 1315,
      "1": 198
    },
    "test_sequence_count": 1513,
    "threshold_source": "train_score_quantile",
    "train_event_count": 14421,
    "train_label_counts": {
      "0": 557
    },
    "train_sequence_count": 557,
    "train_template_vocabulary": 24
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "raw_logs": {
    "path": "/vol/gpudata/hs1822-anomalog/AnomaLog/data/openstack_deeplog_preprocessed/OPENSTACK_DEEPLOG_PREPROCESSED/preprocessed/openstack_labelled_raw.log",
    "sha256": "dad8d1091c5ca9054b230847b689d0b1fea78687b903fe032f2e0f087c2f21bf"
  },
  "run_fingerprint": "690e2c0c7a176fc80472b4f396cec86867b79617925119a6d3e9a856ceccabc6",
  "sequence_config": {
    "grouping": "entity",
    "split": null,
    "step": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "sequence_count": 2070,
  "sequence_split_counts": {
    "ignored": 0,
    "test": 1513,
    "train": 557
  },
  "sequence_split_summary": {
    "effective_train_fraction_of_eligible": 1.0,
    "effective_train_fraction_overall": 0.26908213,
    "eligible_train_sequence_count": 557,
    "excluded_from_train_count": 0,
    "ignored_sequence_count": 0,
    "ineligible_train_pool_count": 0,
    "realised_train_sequence_count": 557,
    "requested_test_fraction": 0.8,
    "requested_train_fraction": 0.2,
    "train_on_normal_entities_only": false,
    "train_pool_sequence_count": 557
  },
  "source": {
    "preset": "openstack_deeplog_preprocessed",
    "type": "preset"
  },
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "structured_parser": "openstack_deeplog_preprocessed",
  "structured_rows": 53618,
  "template_parser": "spell",
  "timestamp_bounds": {
    "max_unix_ms": 1495002306749,
    "min_unix_ms": 1494790742007
  }
}
```


---

## `openstack_deeplog_preprocessed_template_frequency/690e2c0c7a17/metrics.json`

```json
{
  "aggregation_policy": null,
  "available_metric_scopes": [
    "sequence_level_detection"
  ],
  "evaluation_unit": "sequence",
  "ignored_label_counts": {},
  "ignored_sequence_count": 0,
  "label_unit": "sequence",
  "mean_test_score": 3.09960425,
  "metric_blocks": {
    "sequence_level_detection": {
      "abstained_prediction_count": 0,
      "aggregation_policy": null,
      "class_counts": {
        "anomalous": 198,
        "normal": 1315
      },
      "confusion_matrix": {
        "fn": 197,
        "fp": 4,
        "tn": 1311,
        "tp": 1
      },
      "counted_predictions": 1513,
      "diagnostics": {
        "abstain_rate": 0.0,
        "auto_coverage": 1.0,
        "class_counts": {
          "anomalous": 198,
          "normal": 1315
        }
      },
      "evaluation_unit_count": 1513,
      "headline_metrics": {
        "accuracy": 0.86715135,
        "f1": 0.00985222,
        "precision": 0.2,
        "recall": 0.00505051
      },
      "ignored_prediction_count": 0,
      "invalid_reason": null,
      "label_unit": "sequence",
      "metric_scope": "sequence_level_detection",
      "prediction_unit": "sequence",
      "status": "diagnostic_only"
    }
  },
  "prediction_unit": "sequence",
  "primary_metric_scope": "sequence_level_detection",
  "sequence_count": 2070,
  "split_policy": {
    "application_order": null,
    "raw_entry_split": null,
    "raw_entry_split_summary": null,
    "straddling_group_policy": null,
    "test_fraction": 0.8,
    "train_fraction": 0.2,
    "train_on_normal_entities_only": false
  },
  "stream_segment_policy": {
    "mode": "entity_sequence",
    "train_on_normal_entities_only": false
  },
  "test_label_counts": {
    "0": 1315,
    "1": 198
  },
  "test_sequence_count": 1513,
  "train_label_counts": {
    "0": 557
  },
  "train_sequence_count": 557
}
```
