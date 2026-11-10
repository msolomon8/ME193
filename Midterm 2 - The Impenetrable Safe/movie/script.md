# Stop-motion movie script: How my neural net counts fingers

Read each scene's lines while its pictures are on screen. Easiest way: play `neural_net_movie_GUIDE.mp4` and read the highlighted words as they light up.

## 0. Title: The Impenetrable Safe  (0:00–0:17, 18s)

This is the Impenetrable Safe. It only opens when I show it the right three numbers with my fingers. In this video I'll explain how the neural network inside it figures out how many fingers I'm holding up.

## 1. Pixels: A picture is just numbers  (0:17–0:39, 22s)

A webcam looks straight down at the safe, where my hand goes between two red markers. The whole picture is shrunk to 112 by 64 pixels. To the computer, every pixel is just three numbers, for red, green and blue. So the network starts with 21,504 numbers.

## 2. Convolution: Layer 1: convolution  (0:39–1:10, 31s)

The first layer is a convolution. A filter is a small grid of weights. It slides across the image, and at each spot it multiplies the pixels under it by its weights and adds them up. When the patch matches the filter's pattern, like the edge of a finger, the answer is big. My first layer has 16 different filters, so it makes 16 new images called feature maps.

## 3. ReLU: ReLU: keep the positives  (1:10–1:28, 18s)

Next comes ReLU. Every negative number becomes zero, and positive numbers stay the same. This is the same bend we used to fit the V data. Without it, stacking layers would just add up to one big straight line.

## 4. Pooling: Max pooling: shrink it  (1:28–1:47, 20s)

Then max pooling. In every two by two block, only the biggest number survives. The feature maps shrink to half the size, but the strong features stay. It also means a finger that moves a few pixels still gives the same answer.

## 5. Three layers: Repeat three times  (1:47–2:09, 22s)

The network does this three times: convolution, ReLU, pooling. Each time the images get smaller, but there are more of them: 16, then 32, then 64 feature maps. The early filters find simple edges. The later ones find bigger shapes, like fingertips and the gaps between fingers.

## 6. Linear layer: Flatten + linear layer  (2:09–2:31, 22s)

At the end, the 64 maps of 8 by 14 are flattened into one long list of 7,168 numbers. A linear layer connects every one of those numbers to seven outputs, one for each possible answer: zero to five fingers, or blank, when there's no hand at all.

## 7. Softmax: Softmax: the answer  (2:31–2:46, 15s)

Softmax turns the seven scores into probabilities that add up to one hundred percent. The biggest one is the answer. Here the network says three fingers, and it's 92 percent sure.

## 8. Training: How it learns  (2:46–3:18, 32s)

But how does it learn what the filters should be? Training. I showed it hundreds of labeled photos. When it guesses wrong, cross-entropy loss measures how wrong it was. Backpropagation uses the chain rule to find how much each weight caused the error, and gradient descent nudges every weight a little bit downhill. It's the same idea as m and b in my spreadsheet, just with about 49,000 weights instead of two.

## 9. Augmentation: More practice  (3:18–3:36, 18s)

To give it more practice, every training photo is randomly rotated, shifted, flipped and made brighter or darker. The network sees a slightly different version every time, so it learns what fingers look like instead of memorizing the photos.

## 10. The lock: Opening the safe  (3:36–4:01, 25s)

On the safe, I press the button and the lights come on. A digit only counts if the network is sure for a full second, and I have to take my hand away before the next one. After three digits, if they match the secret code, the servo swings the latch and the safe opens.

## 11. End: Thanks for watching!  (4:01–4:08, 7s)

And that's how the Impenetrable Safe counts fingers. Thanks for watching!

**Total: 4:08**