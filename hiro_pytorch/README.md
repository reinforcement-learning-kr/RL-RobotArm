# data-efficient hrl(pytorch)
논문 [data-efficient-hrl](https://arxiv.org/abs/1805.08296) (이하 hiro)을 파이토치로 구현하였습니다.
model-free 알고리즘인 [TD3](https://arxiv.org/abs/1802.09477)을 기반으로 작동되는 코드인 점을 감안하여
논문에 등재된 [공식 TD3 코드](https://arxiv.org/abs/1802.09477)를 기반으로 구현되었습니다([코드](https://github.com/sfujim/TD3)).

### 파일설명
```hiro.py``` : hiro의 구조를 모두 포함하고 있는 메인 파일입니다.  

 ```TD3.py``` : agent를 생성하는 파일입니다. [공식 TD3 코드](https://arxiv.org/abs/1802.09477)의 TD3.py 와 동일합니다.
 
 ```utils.py``` : experience memory를 생성해줍니다.  
 [공식 TD3 코드](https://arxiv.org/abs/1802.09477)에 있는 메모리 클래스와 hiro에 맞게 수정되어진 메모리가 있는데, 
 기본적으로 hiro에 맞게 수정되어진 메모리에 저장을 하고, 추후 train을 할 시에 함수를 통하여 TD3 학습에 맞게끔 데이터를 수정하여 공식 메모리 클래스에 
 임시로 저장을 해 준다음 TD3 학습을 시키게 됩니다.

### 필요 모듈 설치 명령어
#### python2.7
```$ pip install torch torch-vision gym```  
```$ pip install -U 'mujoco-py<1.50.2,>=1.50.1'```

#### python3
```$ pip3 install torch torch-vision gym```  
```$ pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'```

### 실행방법
실행방법은 다음과 같습니다 :  
```$ python hiro.py --env Fetchreach-v1 --render=True ```  

Flag 기본값이 hiro.py에 모두 정의되어있는 만큼 hiro.py를 Flag 없이 실행시켜도 돌아갑니다.
hiro.py 를 참고하시고 변경을 원하시는 상수들은 임의로 바꾸실 수 있습니다.
