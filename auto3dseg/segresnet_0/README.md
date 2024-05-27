# Auto3dseg segresnet AbdomenAtlas Benchmark




# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install monai==1.3.0
```

# Models

Please download the trained weights for Swin UNETR backbone (subject to update the lastest) from this <a href="https://www.dropbox.com/scl/fi/oh3fjp7wce9qfk3tmxxfx/model.pt?rlkey=jnbvduo2arpp8q2xxvd1zprwa&st=lf2oja1y&dl=0"> link</a>.

And move the checkpoint into ./segresnet_0/work_dir/segresnet_0/model/model.pt

## Inference



``` bash
bash run_testing.sh
```
