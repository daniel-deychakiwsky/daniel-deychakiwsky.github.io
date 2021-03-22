---
layout: post
title: Simple Podcast Ad Insertion
categories: digital-signal-processing podcast
author: Daniel Deychakiwsky
meta: Simple Podcast Ad Insertion
mathjax: true
permalink: /:title
---

This post outlines a simple time-domain energy-based
algorithm for identifying ad opportunities in podcast audio.
This post assumes basic knowledge of digital signal processing (DSP).

* TOC
{:toc}

# Context

## Podcasting Company

Imagine you and I start a podcasting platform business.
Users can create and / or consume content (podcasts) on the platform.
Our business model offers a free-tier consumer experience, i.e.,
anyone with an account can listen to any podcast for free.
We decide to monetize our business by inserting audio ads but how
do we know where to interrupt a podcast to play an ad?

## Engineering Problem

We call upon our engineering team to prototype a simple solution for a **next day**
proof-of-concept (POC) demo. To be clear, our acceptance criteria is to 
find the "least obtrusive" ad breaks or insertion points for a given podcast.
We don't need to worry about cross-fading the content with the ad 
for smooth transitions as that will be handled by
a different engineering team. Our focus is solely identifying ad breaks.

## Engineering Approach

### Machine Learning (ML) / Artificial Intelligence (AI)

We may be tempted to apply ML / AI to solve this problem.
While they are attractive options, they
require lots of data in order to learn. We could explore the subset of
unsupervised ML algorithms which don't require additional information
(labels) but even those require substantial time-boxing for tuning 
hyperparameters.

Google's [first rule] of ML is to not use ML; in our case it may
be wise to follow such a philosophy in order to meet the deadline for our POC.
If we convince our stakeholders of a value-add, we can
incorporate ML / AI solutions in future iterations.

> Machine learning is cool, but it requires data.
> Theoretically, you can take data from a different
> problem and then tweak the model for a new product,
> but this will likely underperform basic heuristics.
> If you think that machine learning will give you a 100% boost,
> then a heuristic will get you 50% of the way there.

### Digital Signal Processing (DSP)

It turns out that we can chalk-up a simple algorithm by applying
basic DSP techniques. If you don't know DSP, I'd suggest taking a few
courses to learn it as the applications are endless. Coursera offers
exceptional [foundational] and useful [music applications] courses.

# Solution / Algorithm

## Approach

How would you define "least obtrusive"? This is a subjective measure.
Is a podcast audio ad break best defined by a change in conversation,
a change in sentiment, a change in emotion, the lack of voice activity,
or something else?

In the simplest case, we can develop a time-domain energy-based algorithm which
may translate to one of many possible definitions on a case-by-case basis.

We can track the signal's energy over time and insert ads where the energy is lower
than some threshold for some amount of time (relative to the signal).
We can find several of these insertion
points and order them by the largest amount of time under the threshold.
By this definition, the best ad insertion point is the
maximal contiguous thresholded low energy streak in time.

If you're into stock trading, you may see this algorithm's resemblance to
several technical indicators that you may be familiar with.

### Root Mean Square (RMS)

We will use [RMS] to proxy the de facto time-domain signal [energy] calculation.
They measure different spins of the same thing, the $L^2$ or [euclidean norm]. 

In order to see how RMS changes over time we can
apply a framed version of the calculation yielding a new signal
which captures how the RMS of our signal changes over time.

When I say "framed" I mean we're considering $n$ audio samples
in a single calculation, storing the result, shifting or hopping over by
$m$ samples, and then repeating the process until we traverse the entire signal.

Why not just use a framed version of the energy calculation?
I'm using RMS because a framed version of it is already implemented
in the audio package I'm importing. Work smart, not hard :).

### Amplitude Envelope (AE)

The [AE] is also a framed calculation and is simply the maximum value of a given frame.
Its calculation yields a signal that traces the upper envelope of the original signal over time.
Valerio Velardo's [implementation] is all we need.

```python
import numpy as np

def amplitude_envelope(x, frame_size=2048, hop_length=512):
    return np.array([max(x[i: i + frame_size]) for i in range(0, x.size, hop_length)])
```

### AE + RMS

We combine these two measures to form our final algorithm. We set the
frame and hop sizes that will be shared by both measures and carry out the calculations.
We use the AE as our measure of how the signal's energy changes over time,
and the standard deviation of the framed RMS as the threshold.

Everytime the AE dips below the standard deviation of the framed RMS,
we mark the beginning of a potential ad break segment. Once it breaks back over the threshold,
we mark the end of an ad break segment and add that segment to
a temporary result buffer. Once we've done this for the entire AE, we sort the segments
by the length of the segment, descending. Finally, we define insertion points as the
midpoints of these segments, where the largest segments are considered to be better
ad breaks or insertion points.

## Demo

Here's a fake podcast with its waveform:

{% include embed-audio.html src="/assets/audio/simple_podcast_ad_insertion/fake_podcast.mp3" %}

#### Fig. 1
![fake_podcast]

Here's a fake audio ad with its waveform:

{% include embed-audio.html src="/assets/audio/simple_podcast_ad_insertion/fake_ad.mp3" %}

#### Fig. 2
![fake_ad]

Here's the fake podcast waveform overlaid
with our algorithm's calculations and results: 

#### Fig. 3
![fake_result]

[Fig 3.](#fig-3) shows the time-domain waveform in blue,
the AE in red, RMS in pink, +/- one standard-deviation of
RMS in yellow, and three ad breaks or insertion points in green
which vary in thickness (thicker lines are better ad breaks).

Here's something a bit less basic, a [mel]-[spectrogram] of the fake podcast:

#### Fig. 4
![fake_result_spec]

Comparing [Fig 3.](#fig-3) to [Fig 4.](#fig-4)
notice that the algorithm places ad breaks
where there is a lack of frequency content,
in this case, the harmonics produced by the speaking voice;
the algorithm is choosing to
insert ads where speech is not present.

Here's the result of stitching the fake ad
to the fake podcast in the best of the three ad breaks the
algorithm has suggested and its waveform. This stitching isn't meant to sound good,
it's just a simple cut and paste. Remember, we'd dish this work off
to a different engineering team or tackle it in a different scope.

{% include embed-audio.html src="/assets/audio/simple_podcast_ad_insertion/fake_stitched.mp3" %}

#### Fig. 5
![fake_stitched]

[first rule]: https://developers.google.com/machine-learning/guides/rules-of-ml#rule_1_don%E2%80%99t_be_afraid_to_launch_a_product_without_machine_learning
[foundational]: https://www.coursera.org/learn/dsp1
[music applications]: https://www.coursera.org/learn/audio-signal-processing
[implementation]: https://www.youtube.com/watch?v=rlypsap6Wow
[euclidean norm]: https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
[energy]: https://en.wikipedia.org/wiki/Energy_(signal_processing)
[RMS]: https://en.wikipedia.org/wiki/Root_mean_square#Definition
[AE]: https://en.wikipedia.org/wiki/Envelope_(waves)
[spectrogram]: https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,they%20may%20be%20called%20waterfalls.
[mel]: https://en.wikipedia.org/wiki/Mel_scale#:~:text=The%20mel%20scale%20(after%20the,dB%20above%20the%20listener's%20threshold.

[fake_podcast]: assets/images/simple_podcast_ad_insertion/fake_podcast.png
[fake_ad]: assets/images/simple_podcast_ad_insertion/fake_ad.png
[fake_stitched]: assets/images/simple_podcast_ad_insertion/fake_stitched.png
[fake_result]: assets/images/simple_podcast_ad_insertion/fake_result.png
[fake_result_spec]: assets/images/simple_podcast_ad_insertion/fake_result_spec.png
