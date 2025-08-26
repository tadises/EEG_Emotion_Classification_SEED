# EEG_Emotion_Classification_SEED
Using SEED dataset,  CNN with SE module and a Transformer for temporal encoding, achieve an acuraccy of 94.77 ± 0.34 %

models.py is the original net design from [patrickdmiller/SEED-EEG-Deep-neural-network: residual deep cnn and lstm for classifying SEED data](https://github.com/patrickdmiller/SEED-EEG-Deep-neural-network)
modeltest.py is the new modified net design. Added SE module in deep cnn and replace lstm as a transformer module

![image](pictures/图3.1.png)![image](pictures/图4.3(a).png)![image](pictures/图4.3(b).png)![image](pictures/图4.3(c).png)

![image](pictures/图4.4(a).png)

![image](pictures/图4.5(a).png)
