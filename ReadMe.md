Image caption work in lua. <br />
On the basis of Torch7 framework. <br />
Under Development. <br />

### Requirements
1. torch7 http://torch.ch/docs/getting-started.html <br />
2. several packages besides default torch7:<br />
    (1) dp:	luarocks install dp<br />
    (2) fs:	luarocks install fs<br />
    (3) rnn:	luarocks install rnn<br />
    (4) hdf5:	luarocks install hdf5<br />


### Steps

#### 1. Prepare your data 
##### 1) Run TorchPrepare.py <br />
This file reads the raw sentences, and prepares them into json files to be further processed
##### 2) Run prepo.py <br />
This file converts the json files into h5 file, to be further read by lua script

TODO: 1)intergrate TorchPrepare.py and prepo.py into one single file, as preprocessing<br />
