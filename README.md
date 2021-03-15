# Imitation Learning with CarRacing-v0

## Behavioural Cloning
There are three major strands of imitation learning: behavioural cloning, direct policy learning, and inverse reinforcement learning. 
Behavioural cloning is by far the simplest - the agent is a supervised learning algorithm mapping states to actions. The agent aims to minimise the distance between the agent and the expert’s behaviours.

Direct policy learning is a similar algorithm, but requires more expert data. When the agent becomes uncertain, it will consult the expert for guidance. This type of algorithm is particularly useful in safety-critical environments, where uncertainty could lead to undesired risky behaviour. It is also well-suited to environments where a wrong move could take the agent to a state significantly different from those visited by the expert, and therefore seen before by the agent. In these situations, without guidance from the expert the agent would be acting at random. 

Inverse reinforcement learning algorithms are two-staged. In the first stage, the agent observes training data and attempts to learn the reward function the expert was trying to maximise. Then, a classic reinforcement learning algorithm can be used to learn behaviour that will maximise this reward function.

Of the three, behavioural cloning stands out for its simplicity. It doesn’t require access to the environment, nor the reward function or an interactive human expert who can help correct its errors. Furthermore, a racing car game is a very low risk environment where mistakes are immaterial. Wrong moves that take the agent into totally novel states are unlikely as the state space is so small and simple, making direct policy learning an overkill. In a higher risk environment, I would certainly consider a more complex algorithm, however for this case study behavioural cloning seems the most appropriate.  

## Model Choice
I chose a convolutional neural network (CNN) to model the data. CNNs are neural networks that include convolutional layers that are particularly good for identifying features in image data. As each state in the racing game is a 97x97 pixel image, a CNN seemed the most appropriate choice. I built a fairly shallow CNN with only 4 convolutional layers, as the images are very simple. 

I decided to stack together 4 of the environment states into one stacked-state and feed this as input to the model. The reason I chose to do this is that there is no measure of velocity in the game, and so the agent would have no capacity to understand why the “expert” was accelerating or braking depending on the current speed. By stacking 4 frames together, the agent is given a notion of velocity. 

Each state was represented by a small RGB image, which I chose to convert to greyscale before inputting to my model. I made this choice to simplify the input image as, with such a simple input image, I thought little information would be lost. Furthermore, the combination of colour channels and velocity channels may have required a more sophisticated CNN.

My last significant model choice was to remove 25% of state action pairs where no action was taken. The choice to remove some of the imbalance in the data was to speed up learning time. 

## Results

![car_demo](https://user-images.githubusercontent.com/9541955/111087782-c75c5a00-8523-11eb-9b8f-030bf9cb9728.gif)

Above is a gif of my trained agent moving in a completely straight line. This sub-optimal behaviour could be due to a few reasons - insufficient training time, insufficient or poor quality expert data, or a mistake in the model I’ve not noticed. I think the most likely explanation is that there are very few data where the “expert” turns left or right. This could be addressed by augmenting turning training examples or collecting more data. Also, turning the image to greyscale may have made the state action pairs harder to learn. A future improvement could be to include colour channels.

Loss of the trained model over 10 epochs:

![loss_plot](https://user-images.githubusercontent.com/9541955/111122816-d9171f00-856e-11eb-9f4e-927b147a3d7f.png)


