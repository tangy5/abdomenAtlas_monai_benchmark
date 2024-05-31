# Auto3dseg segresnet AbdomenAtlas Benchmark




# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install monai==1.3.0
```

# Models

Please download the trained weights (subject to update the lastest) from this <a href="https://www.dropbox.com/scl/fi/zs57tj3ej5jv4kreeerzv/model_auto3dseg_segresnet_s1.pt?rlkey=wfddlav6lf4vpxtl68h8e7dn1&st=4gvdllwa&dl=0"> link</a>.

And move the checkpoint into ./segresnet_0/work_dir/segresnet_0/model/model.pt

## Inference



``` bash
bash run_testing.sh
```
