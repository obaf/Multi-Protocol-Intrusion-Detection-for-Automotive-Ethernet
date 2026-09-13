# data/

This directory is intentionally empty in the repository. The two datasets are
distributed by their original authors and must be downloaded separately; the
main README ("Data" section) gives the sources, licences and the exact layout
the build scripts expect. After download and `python src/build_*.py` the
directory looks like this:

```
data/
  towids_extracted/
    Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap   (198,979,326 bytes)
    Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap       (134,309,632 bytes)
    y_train.csv                                                          (26,326,022 bytes)
    y_test.csv                                                           (17,173,503 bytes)
  someip_data/
    Error_on_error/      error_on_error_<i>_x.pickle, error_on_error_<i>_y.pickle      (35 captures)
    Error_on_event/      error_on_event_<i>_x.pickle, ...                              (43 captures)
    Missing_request/     missing_request_<i>_x.pickle, ...                             (40 captures)
    Missing_response/    missing_response_<i>_x.pickle, ...                            (40 captures)
  towids.npz             built by src/build_towids.py            (1,995,348 x 119 features)
  someip.npz             built by src/build_someip.py            (2,374,996 x 119 features)
  towids_ts_train.npy    built by src/extract_towids_timestamps.py
  towids_ts_test.npy     built by src/extract_towids_timestamps.py
```
