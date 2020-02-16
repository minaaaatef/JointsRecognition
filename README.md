# joints-action-recognition

## Tutorial For Model Training
we will connect to the training servers vie ssh 
The ip to the server will be provied soon

### prerequisite
1. Linux terminal 
	for windows install [this](https://www.howtogeek.com/336775/how-to-enable-and-use-windows-10s-built-in-ssh-commands/)
2. Internet connection

## connecting to server
* use ssh
 ```python
 ssh root@<ip addres> 
```
* clone rebo
```python 
git clone https://github.com/minaaaatef/JointsRecognition
```
```python
cd JointsRecognition
```
* update server packages 
```python 
apt update 
```
* install pip3
```pytohn
apt install python3-pip
```
* install prerequisite
```python 
pip3 install setuptools
pip3 install -r requirements
```
* Run training.py
```python
python3 Lstm.py
```


## using tmux to make the training running in the background 
Now we were able to run the training but it's runing in out terminal. if we closed the terminal or the connection is lost will have to restart for the beginig 
to avoid this we will use tmux

```pytohn 
apt install tmux
```

* to start a tmux session 
``` pythyon 
tmux
```

* Then run the trainig as before 

now if you closed the terminal you could reattach to the session by 
```python 
tmux attach
```


