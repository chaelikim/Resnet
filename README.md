# Resnet 을 사용한 이미지 분류 학습
### Pytorch의 모듈 중 resnet을 이용해서 대권주자 얼굴을 인식하여 분류해주는 학습을 해보겠습니다.

우선 대선후보의 사진을 크롤링할 것입니다.
https://github.com/YoongiKim/AutoCrawler 를 참고하여 셀레니움으로 웹크롤링 하였습니다.
이용방법은 다음과 같습니다. 

### 
    1. https://github.com/YoongiKim/AutoCrawler 링크에서 오토크롤러를 다운로드 후 실행합니다.
    2. Keywords.txt에 크롤링 하려는 검색어를 한줄에 하나씩 씁니다.
    3. 터미널 창에서 >>python main.py를 실행시킵니다.
    4. 사진과 같이 download 폴더에 크롤링된 사진이 저장된 것을 확인할 수 있습니다.


![1](https://user-images.githubusercontent.com/91925500/135967339-bc168828-0034-4791-9664-30af59af6a8c.png)![2](https://user-images.githubusercontent.com/91925500/135967370-f6b59d7c-53b2-4157-851e-6a668a065fa1.png)
![3](https://user-images.githubusercontent.com/91925500/135967580-25f99b40-ed80-4ff9-899e-0d2ee537d2eb.png)


### resnet을 이용해서 이미지 분류 및 전이학습을 시킵니다.
전이학습하는 소스는 다음 사이트를 참고하였습니다.
https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-pytorch
### 전이학습한 코드로 응용하기
* 먼저 딥러닝 학습에 필요한 여러 라이브러리를 import 합니다. 특히 torchvision의 models를 import하여 먼저 resnet50모델을 사용할 수 있도록 해야 합니다.
~~~python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
~~~

* GPU가 있으면 GPU로 학습하도록 설정합니다.
~~~python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
~~~

* resnet50 모델을 가져옵니다. pretrained는 ImageNet으로 사전 학습된 모델을 가져 올지를 결정하는 패러미터입니다. 우리는 True를 설정합니다.
또한 미리 학습된 모델로 finetuning 하는것이므로 requires_grad = False로 설정해 주어야 학습이 안 되도록 고정시킬 수 있습니다. 불러온 모델의 마지막 fc(fully connected) layer를 수정하여 fc layer를 원하는 레이어로 변경한다. 출력이 5명이 되도록 분류하는 모델을 만들 것이므로 nn.Linear(128,5)를 사용한다.
~~~python
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 4)).to(device)
model.load_state_dict(torch.load('./weights.h5'))
~~~

* 크롤링한 대선 후보의 사진들을 resnet의 입력에 적합하도록 트랜스폼 하는 함수를 만듭니다.
  * transforms.RandomAffine(degrees) - 랜덤으로 affine을 변형한다.
  * transforms.RandomHorizontalFlip() - 이미지를 랜덤하게 수평으로 뒤집는다.
  * transforms.ToTensor() - 이미지 데이터를 텐서로 바꿔준다.
  * transforms.Nomalize(mean,std,inplace=False) -이미지를 정규화한다.
~~~python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),normalize ]),
}
~~~

* DataLoader를 사용하여 이미지들을 읽습니다.
~~~python
image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
}
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}
~~~

* 손실함수는 CrossEntropyLoss, 옵티마이저는 Adam을 사용하도록 설정합니다.
~~~python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())
~~~

* 이미지를 학습하는 함수를 작성합니다. 일반적인 파이토치 학습 코드와 동일합니다.
~~~
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model
    ~~~
* 모델을 학습합니다. 학습하는 시간이 조금 걸립니다.
~~~python
model_trained = train_model(model, criterion, optimizer, num_epochs=3)
~~~

* 학습이 완료된 모델을 저장합니다.
~~~python
torch.save(model_trained.state_dict(), './weights.h5')
~~~

* 모델을 다시 만듭니다. 이번에는 학습을 하지 않고 저장된 모델을 로드할것이라서 프리트레인을 False로 설정합니다. 위에서 학습한 weight값을 읽어 모델을 준비합니다.
~~~python
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 4)).to(device)
model.load_state_dict(torch.load('./weights.h5'))
~~~

* 테스트할 이미지를 준비합니다.
~~~python
validation_img_paths = ["validation/윤석열/google_0008.jpg",
                        "validation/이재명/naver_0006.jpg",
                        "validation/추미애/google_0013.jpg",
                        "validation/홍준표/google_0007.jpg"]
img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
~~~

* 이미지를 resnet50에 적합한 입력으로 만들기 위해 트렌스폼합니다.
~~~python
validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])
                                ~~~
* 학습된 모델로 테스트 이미지를 예측합니다. 예측된 결과는 softmax를 사용하여 어떤 이미지로 분류되는지 확률을 보여줍니다.
~~~python
pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
~~~
* matplotlib에서 한글이 깨지는 것을 방지하기 위한 처리를 합니다.
~~~python
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font= font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
~~~

* 테스트 사진이 어떤사람의 사진인지 화면에 출력해 봅니다. argmax함수를 통해 가장 높은 사람의 인덱스를 구해서 해당 이미지와 이름을 출력합니다.
~~~python
labels = ['윤석열', '이재명', '추미애','홍준표']
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title(labels[np.argmax(pred_probs[i])])
    ax.imshow(img)
~~~
* 테스트 이미지를 학습한 모델로 분류해봅니다. 아주 잘 작동합니다.
![4](https://user-images.githubusercontent.com/91925500/135976367-c224657b-d43e-43bf-9803-97ff52b27988.PNG)

