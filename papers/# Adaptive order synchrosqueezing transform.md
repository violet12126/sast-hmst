# Adaptive order synchrosqueezing transform

Marcelo A. Colominas <sup>a,∗</sup>, Sylvain Meignen <sup>b</sup> 

![](images/459cb3dba1df89d87e88115d1554ea50bf0706e742d48cbfb5bf3b6ebe1e78e0.jpg)


<sup>a</sup> Institute for Research and Development in Bioengineering and Bioinformatics (CONICET - UNER), and Faculty of Engineering (UNER), Oro Verde, 3100, Entre Rios, Argentina 

<sup>b</sup> Jean Kuntzmann Laboratory- Bâtiment IMAG, Université Grenoble Alpes, Saint Martin d’Hères, 38401, France 

## A R T I C L E I N F O

Keywords: Time–frequency Synchrosqueezing transform Multicomponent signals 

## A B S T R A C T

Non-stationary signals are characterized by time-varying amplitudes and frequencies. Tracking them is important for studying the dynamic systems that generate the signals, the synchrosqueezing transform (SST) being a versatile and widely used tool for such a task. In this paper, we address the problem of locally selecting the order for SST, which can be difficult in the presence of strong modulations and noise. We propose to tackle this problem by minimizing the Rényi entropy to maximize the concentration on the time–frequency plane. We do that using coordinate descent, and sparse matrices. Results show superior representations to those obtained with fixed order SST, both in terms of concentration and error with respect to the ideal representation. We illustrate the capabilities of our proposal on real-world signal with strong frequency modulation: bat social vocalization, gibbon song, and voice signal 

## 1. Introduction

Many non-stationary signals such as audio signals (music, speech, bird songs, . . . ) [1], electrocardiogram [2], and thoracic and abdominal movement signals [3] can be approximated as a superimposition of amplitude and frequency-modulated (AM/FM) modes, called a multi component signal (MCS). To estimate frequency variations over time is essential when dealing with MCSs [4], for which the short-time Fourier transform (STFT) is commonly used, and also the spectrogram, the squared modulus of the STFT, for visualization purposes. The quality of estimation is however tightly related to the choice of the window since, due to Heisenberg uncertainty principle, the spectrogram smears the information in the time–frequency (TF) plane. To improve that aspect, techniques based on the adaptive short-time Fourier transform [5] were developed. The computational cost of such transforms compared with STFT is however considerably increased. Another limitation is that the adaptation of the window length in the STFT can only be done timewise, which is a concern when the signal contains different modes with different frequency modulations at a given time. 

Another strategy to compensate for the smearing of the information in the TF plane when using STFT is to consider a reassignment technique like synchrosqueezing transform (SST), first introduced in the wavelet context [6] and then extended to STFT in [7,8]. Though such a reassignment process is efficient when the modes are slightly modulated, it is inaccurate when this hypothesis is not verified, therefore new techniques were developed to better take into account this stronger modulation. First, a linear chirp model was proposed for the modes making up the MCS [9,10] and then to take into account fast oscillating phases, this model was extended [11] through the so-called high order SST. It is worth noting here that a bunch of works has been proposed to generalize the SST by mixing adaptive Fourier transform and SST as in [12,13], and by considering chirplet transform and SST [14–16]. The interest of the latter group of techniques is that they can cope with crossing-over components. 

Nevertheless, letting apart this issue of crossing-over components, the main limitation of these reassignment processes is that it is very complicated to figure out how they deal with noise. For instance, even when the phase of a mode presents some strong oscillations, it is not obvious that a high order SST will lead to a better TF representation (TFR). For that reason, different techniques were recently developed to locally adapt the order of SST in the TF plane, one based on maximizing the energy on the ridge associated with the modes making up the MCS [17], and another one aiming at minimizing the Rényi entropy (therefore maximizing the concentration) by moving the coefficients one by one [18], not requiring that some ridges are detected beforehand. The problem of such a technique, is that it is slow, and not presented in a clear optimization framework. Our goal in this paper is to work on these two aspects. For that purpose, we present in Section 2 synchrosqueezing techniques. In Section 3, after having explained why it is necessary to adapt the order of reassignment in the presence of noise, we introduce our novel adaptive SST algorithm based on a solid optimization framework. In Section 4, we first show on simulated signals that the adaptive SST we propose achieves better concentration than SST with fixed order, which results in better estimation of the amplitude of the modes. This remains true when dealing with real signals, where we notice that the value on the ridges of the reassigned transform with adaptive SST is much more stable than when SST with a fixed order is used. Regarding applications, compared with classical SSTs, the proposed adaptive SST offers a better readability of the time– frequency representation of different types of vocal and ECG signals, and improves amplitude estimations for that type of signals. 

## 2. Synchrosqueezing transforms

In this section, we introduce a series of definitions we use through out the paper. 

## 2.1. Short-time Fourier transform

Considering a signal $x \in L ^ { 1 } ( \mathbb { R } ) \cap L ^ { 2 } ( \mathbb { R } )$ and a real window $g \in$ $L ^ { 1 } ( \mathbb { R } ) \cap L ^ { 2 } ( \mathbb { R } )$ , the (modified) Short-Time Fourier Transform (STFT) is defined as: 

$$
F _ {x} ^ {g} (t, \eta) = \int_ {\mathbb {R}} x (u) g (u - t) e ^ {- 2 i \pi (u - t) \eta} d u,\tag{1}
$$

where $t \in \mathbb { R }$ stands for time, and $\eta \in \mathbb R$ stands for frequency. 

The signal ?? can be reconstructed with the vertical reconstruction formula 

$$
x (t) = \frac {1}{g (0)} \int_ {\mathbb {R}} F _ {x} ^ {g} (t, \eta) d \eta .\tag{2}
$$

When dealing with noisy signals $\tilde { x } \ = \ x + \xi ,$ , where ?? is a white Gaussian noise, signal denoising can be performed through: 

$$
x (t) \approx \frac {1}{g (0)} \int_ {\Gamma} F _ {\tilde {x}} ^ {g} (t, \eta) d \eta \approx \frac {1}{g (0)} \int_ {\mathbb {R}} \tilde {F} _ {\tilde {x}} ^ {g} (t, \eta) d \eta ,\tag{3}
$$

where ${ \varGamma } = \{ ( t , \eta ) / | F _ { \tilde { x } } ^ { g } ( t , \eta ) | > 3 \gamma \}$ 1 , and 

$$
\tilde {F} _ {\tilde {x}} ^ {g} (t, \eta) = \left\{ \begin{array}{l} F _ {\tilde {x}} ^ {g} (t, \eta), \text {if} | F _ {\tilde {x}} ^ {g} (t, \eta) | > 3 \gamma \\ 0, \text {otherwise}, \end{array} \right.\tag{4}
$$

so that the reconstruction can be carried out either by limiting the integration domain or by using the thresholded STFT. Indeed, $\tilde { F } _ { \tilde { x } } ^ { g } ( t , \eta )$ constitutes a thresholded version of the STFT, and using it for reconstruction performs a denoising task. The threshold ?? is estimated as $\tilde { \gamma } = \sqrt { 2 } \mathrm { m e d i a n } ( | \Re \{ F _ { \tilde { x } } ^ { g } ( t , f ) \} | ) / 0 . 6 7 4 5 ~ [ 1 9 \mathrm { - } 2 1 ]$ 

## 2.2. Synchrosqueezing and high-order versions

Synchrosqueezing was originally proposed within the wavelet con text $[ 6 , 2 2 ]$ . Adapted to the STFT setting $[ 8 , 2 3 ]$ , it aims at ‘‘sharpening’’ the STFT by vertically reassigning the coefficients: 

$$
S _ {x} ^ {g, N} (t, \eta) = \frac {1}{g (0)} \int_ {\mathbb {R}} F _ {x} ^ {g} (t, v) \delta (\eta - \tilde {\omega} _ {x} ^ {[ N ]} (t, v)) d v,\tag{5}
$$

so that the new TFR is ‘‘closer’’ to the ideal one. Now ?? can be either a noiseless or noisy signal (in which case a thresholding can be performed on the STFT), and $\tilde { \omega } _ { x } ^ { [ N ] } ( t , f )$ is the ??th order IF estimation built assuming a phase that locally behaves as a polynomial of order ?? [11]. In what follows, SSTN denotes the synchrosqueezing of order ??. The case of $N = 1$ was addressed in [8,23]; $N = 2$ was proposed in [9]; and $N \geq 3$ was discussed in [11], giving birth to the so-called high-order synchrosqueezing. The estimation of the IF, with an ??th order approximation and using a Gaussian window $g ( t ) = e ^ { - \frac { \pi t ^ { 2 } } { \sigma ^ { 2 } } }$ , can be done with two matrices of STFTs, 

$$
D ^ {[ N ]} (t, \eta) = \left[ \begin{array}{c c c c} F _ {x} ^ {g} & F _ {x} ^ {t g} & \dots & F _ {x} ^ {t ^ {N - 1} g} \\ F _ {x} ^ {t g} & F _ {x} ^ {t ^ {2} g} & \dots & F _ {x} ^ {t ^ {N} g} \\ \vdots & \vdots & \ddots & \vdots \\ F _ {x} ^ {t ^ {N - 1} g} & F _ {x} ^ {t ^ {N} g} & \dots & F _ {x} ^ {t ^ {2 (N - 1)} g} \end{array} \right],\tag{6}
$$

and 

$$
U ^ {[ N ]} (t, \eta) = \left[ \begin{array}{c c c c} 0 & F _ {x} ^ {t g} & \dots & F _ {x} ^ {t ^ {N - 1} g} \\ F _ {x} ^ {g} & F _ {x} ^ {t ^ {2} g} & \dots & F _ {x} ^ {t ^ {N} g} \\ \vdots & \vdots & \ddots & \vdots \\ (N - 1) F _ {x} ^ {t ^ {N - 2} g} & F _ {x} ^ {t ^ {N} g} & \dots & F _ {x} ^ {t ^ {2 (N - 1)} g} \end{array} \right],\tag{7}
$$

where $F _ { x } ^ { t ^ { n } g }$ stands for the STFT with window $t ^ { n } g ( \cdot )$ , and where we omitted (??, ??) for the sake of readability. Indeed, the IF estimation can be written as [24] 

$$
\tilde {\omega} _ {x} ^ {[ N ]} (t, \eta) = \eta - \frac {1}{2 \pi} \Im \left\{\frac {d e t (U ^ {[ N ]} (t , \eta))}{d e t (D ^ {[ N ]} (t , \eta))} \right\}.\tag{8}
$$

It is obvious that as soon as ?? increases so does the number of STFTs involved, and the number of terms and products between different STFTs. In the noisy case, this intensifies the damage caused by noise, worsening the estimations. Here we can see some examples on monocomponent FM signals (see Fig. 1). In all the three cases we computed IF estimations for orders $N = 1 , \ldots , 4$ and different TF locations (fixed time, different frequencies, for a given signal): $\tilde { \omega } _ { x } ^ { [ N ] } ( t _ { 0 } , \eta )$ . We studied the behavior of the IF estimations close to the actual IF (one Hertz from the $\mathrm { I F } ,$ referred to as $\eta = a ,$ where $a = \phi ^ { \prime } ( t _ { 0 } ) { + } 1 )$ , and far from the IF (ten Hertz, $\eta = b$ in our figure, where $b = \phi ^ { \prime } ( t _ { 0 } ) + 1 0 )$ . We show the results in the form of 3D boxplots of the absolute errors $| \phi ^ { \prime } ( t _ { 0 } ) - \tilde { \omega } _ { x } ^ { [ N ] } ( t _ { 0 } , \eta ) |$ for 60 realizations of noisy signals at 10 dB of SNR. 

For the linear chirp case (left column) the results are consistent with the expectation, since the best results are those for $N = 2 ,$ , both close to the IF $( \eta = a )$ and far from the IF $( \eta = b )$ . When far from the IF, $N = 3$ produces the worst approximations. 

The approximation results for an exponential chirp are shown on the middle column: when one is close to the IF, i.e. $\begin{array} { r } { \eta = a , N = 1 } \end{array}$ produces the best results, but far from $\mathbf { i t } ,$ i.e. $\eta ~ = ~ b , ~ N ~ = ~ 2$ leads to the best performance. 

A sinusoidal chirp, along with IF approximation results, are shown on the right column. As expected, $N = 4$ is the best when close to the IF, i.e. $\eta = a ,$ . But when moving away from the IF, i.e. $\eta = b , N = 1$ is the best, with the performance worsening as soon as ?? increases. 

The results just discussed evidence that the selection of the best synchrosqueezing order is far from trivial, as it depends on the frequency modulation, the noise level, and the distance to the actual IF. As a general trend, the estimations worsen with the distance to the actual IF and when the noise level is increased. Therefore, we will discuss an adaptive order selection strategy in the next section. 

## 2.3. Discrete time

Notation. We will use uppercase bold letters ?? to indicate matrices, and $\mathbf { M } [ m , n ]$ will be the coefficient corresponding to the ??−th row and ??th column. Lowercase bold letters ?? will denote column vectors, with ??[??] representing the ??th coefficient. 

Note that the definitions of the previous subsection can be directly extended to the discrete time and frequency setting. Indeed, we can define the matrix 

$$
\mathbf {F} _ {x} ^ {g} [ m, n ] \approx F _ {x} ^ {g} \left(\frac {n}{L}, m \frac {L}{M}\right),\tag{9}
$$

such that 

$$
\mathbf {F} _ {x} ^ {g} = \left[ \begin{array}{c c c c c c} | & | & & | & & | \\ \mathbf {f} _ {1} & \mathbf {f} _ {2} & \ldots & \mathbf {f} _ {n} & \ldots & \mathbf {f} _ {L} \\ | & | & & | & & | \end{array} \right],\tag{10}
$$

where ?? is the length of the signal, ?? the number of frequency bins and $L / M$ the frequency resolution. Here, each column vector $\mathbf { f } _ { n }$ has exactly ?? samples. Equivalently, the thresholded STFT is $\tilde { \mathbf { F } } _ { \tilde { x } } ^ { g } [ m , n ] ,$ with columns $\tilde { \mathbf { f } } _ { n } ,$ for $n = 1 , \ldots , L$ . In this context, the discrete FSSTN reads 

$$
\mathbf {S} _ {x} ^ {g, N} [ m, n ] = \frac {1}{g (0)} \sum_ {q = 0} ^ {M - 1} \tilde {\mathbf {F}} _ {\tilde {x}} ^ {g} [ q, n ] \chi [ m - \lfloor \frac {M}{L} \tilde {\omega} _ {\tilde {x}} ^ {[ N ]} [ q, n ] \rceil ],\tag{11}
$$

![](images/b7fd13c975a7c5c2f283967380e9d5959bd5cb0583a4f9e551133ea5345021d1.jpg)


![](images/2c64f5a7ea08c19b12411028a6657ac2855f7a43f3c9d9356eee55f5a9bac6ae.jpg)


![](images/b9b9a8983d399c4f646cf0c25f6a53f1f44508e581d4b4c41f3a7d7b7faed64c.jpg)


![](images/042be0630a7f8db1952dcea094d01f2aeea3332d280cd7e2a0ab6ae5cb1ffd41.jpg)


![](images/a87e4b960707d97756c8b39129045fad60eb57f8e4f832861af98b5057d3b4f4.jpg)


![](images/db801dfd9cb19d8518f2da4be6721a526aaa612a914b9ae0638953991c05afca.jpg)



Fig. 1. IF estimations for monocomponent FM signals. The results are for 60 realizations of noisy signals at 10 dB of SNR. Left column: linear chirp. Middle column: exponential chirp. Right column: sinusoidal chirp.


where 

$$
\chi [ c ] = \left\{ \begin{array}{l} 1, \text {   if   } c = 0 \\ 0, \text {   otherwise }, \end{array} \right.\tag{12}
$$

and ⌊⋅⌉ stands for the round function (replacing by the nearest integer). The former definition leads to the following implementation of the synchrosqueezing transform: 

$$
\mathbf {S} _ {x} ^ {g, N} [ \lfloor \frac {M}{L} \tilde {\omega} _ {\tilde {x}} ^ {[ N ]} [ m, n ] \rceil , n ] \leftarrow \mathbf {S} _ {x} ^ {g, N} [ \lfloor \frac {M}{L} \tilde {\omega} _ {\tilde {x}} ^ {[ N ]} [ m, n ] \rceil , n ] + \tilde {\mathbf {F}} _ {\tilde {x}} ^ {g} [ m, n ],\tag{13}
$$

for $n = 0 , 1 , \ldots , L - 1$ , and $m = 0 , 1 , \ldots , M - 1 .$ This means that each coefficient $\mathbf { F } _ { \tilde { x } } ^ { g } [ m , n$ ] is reassigned according to $\tilde { \omega } _ { \tilde { x } } ^ { [ N ] } [ m , n ]$ 

The synchrosqueezed representation results in the matrix 

$$
\mathbf {S} _ {x} ^ {g, N} = \left[ \begin{array}{c c c c c c} | & | & & | & & | \\ \mathbf {s} _ {1} ^ {[ N ]} & \mathbf {s} _ {2} ^ {[ N ]} & \ldots & \mathbf {s} _ {n} ^ {[ N ]} & \ldots & \mathbf {s} _ {L} ^ {[ N ]} \\ | & | & & | & & | \end{array} \right],\tag{14}
$$

where the columns ${ \bf s } _ { n }$ have, as before, exactly ?? samples. 

This reassignment process can be viewed as a matrix multiplication as, for each column ${ \bf s } _ { n } ,$ one can write 

$$
\mathbf {s} _ {n} = \mathbf {A} _ {n} ^ {[ N ]} \tilde {\mathbf {f}} _ {n},\tag{15}
$$

where the square matrix 

$$
\mathbf {A} _ {n} ^ {[ N ]} = \left[ \begin{array}{c c c c c c} | & | & & | & & | \\ \mathbf {a} _ {1} ^ {n, [ N ]} & \mathbf {a} _ {2} ^ {n, [ N ]} & \ldots & \mathbf {a} _ {m} ^ {n, [ N ]} & \ldots & \mathbf {a} _ {M} ^ {n, [ N ]} \\ | & | & & | & & | \end{array} \right],\tag{16}
$$

has columns $\mathbf { a } _ { m } ^ { n , [ N ] }$ such that 

$$
\mathbf {a} _ {m} ^ {n, [ N ]} [ p ] = \left\{ \begin{array}{l} 1, \text {   if   } p = \lfloor \frac {M}{L} \tilde {\omega} _ {\tilde {x}} ^ {[ N ]} [ m, n ] \rceil \text {   and   } | \tilde {\mathbf {f}} _ {n} [ p ] | > 0 \\ 0, \text {   otherwise.   } \end{array} \right.\tag{17}
$$

In this setting, it is clear that each coefficient $\tilde { { \bf F } } _ { \tilde { x } } ^ { g } [ m , n ] ~ = ~ \tilde { { \bf f } } _ { n } [ m ]$ is reassigned according to the column $\mathbf { a } _ { m } ^ { n , [ N ] }$ . As each coefficient is reassigned to a unique position, ${ \bf A } _ { n } ^ { [ N ] }$ contains only one non-zero value per column. 

This formalization of synchrosqueezing as a product between a sparse matrix and a vector (the STFT column) allowed for a significant decrease in computational time with respect to the results in [18]. Indeed, the sparsity of matrix ${ \bf A } _ { n } ^ { [ N ] }$ enables the permutations of the columns to be done efficiently. Moreover, this paves the way to parallelization, since the permutations of the columns can be done in parallel, and then one proceeds with the product with the STFT column, and finds out the one with the lowest Rényi entropy. This aspect, however, is out of the scope of the present work. 

## 3. Adaptive order synchrosqueezing

## 3.1. The necessity of an adaptive order

The results shown in Fig. 1 make it evident that the best synchrosqueezing order depends on the frequency modulation, SNR, and the distance to the actual IF. So, to use a fixed order does not guarantee the best possible result. We can think of an adaptive order algorithm for which the synchrosqueezing order depends on each TF point, i.e. to look for a function ??[??, ??]. 

Previous works aimed at defining an adaptive order for the synchrosqueezing transform. In [25] for instance, the selected order was a function of both time and mode number. With such an approach, the order is no longer fixed, but still requires the a priori knowledge of the TF ridges associated with each mode, which is not always available, in particular when the signal contains crossing or intermittent modes, or a high noise level. 

A completely adaptive order was proposed in [18], where the selected order was a function of both time and frequency. Here, each TF point is assigned with an order that maximizes the concentration of the TFR. This rather heuristic approach initializes each TF column with a fixed order, and runs a single cycle of optimization for that column. Our present proposal is funded on [18], but with significant differences: (i) a formalization of the search for the optimal ?? in the form of products of matrices and vectors is introduced; (ii) this formalization allows seeing the problem as a coordinate descent optimization, of which several cycles can be performed; (iii) the initialization step adapts the order to the different subsets, and thus the initial order is no longer fixed for a given column. 

From the previous section, we can see that the order depends, for each coefficient to be reassigned, on the column $\mathbf { a } _ { m } ^ { n , [ N ] }$ . If the orde is now a function of both time and frequency, then the reassignment matrix is ${ \bf A } _ { n } ^ { [ { \bf n } ] }$ , where ?? is a vector of exactly ?? samples (indexed by ??). Then, the columns of ${ \bf A } _ { n } ^ { [ { \bf n } ] }$ are now $\mathbf { a } _ { m } ^ { n , [ \mathbf { n } [ m ] ] }$ 

As for the orders of synchrosqueezing, we have up to $N _ { m a x }$ options $( N \in \mathcal { N }$ where $\mathcal { N } = \{ 1 , \dots , N _ { m a x } \} .$ , where typically $N _ { m a x } = 4 ^ { 1 } )$ . This means that for each column $\mathbf { a } _ { m } ^ { n , [ \mathbf { n } [ m ] ] }$ we have up to $N _ { m a x }$ possibilities. If we have, for a given column of our thresholded STFT $\tilde { \mathbf { f } } _ { n } , \boldsymbol { Q } _ { n }$ non-zero coefficients to be reassigned, then we have up to $( N _ { m a x } ) ^ { Q _ { n } }$ combinations for that single STFT column. For $N _ { m a x } = 4$ and $Q _ { n } = 1 0 ,$ we have more than one million combinations to test for a single STFT column (but we usually have more than 10 non-zero coefficients for an STFT column). With this in mind, the question is how to choose a good combination? 

## 3.2. Measures of concentration

A widely used method to measure the TF concentration is the Rényi entropy [26]. As was done before [18], we will try to minimize (maximize) the Rényi entropy (concentration) of each reassigned column: 

$$
R E _ {1 D} (\mathbf {v}) = \frac {1}{1 - \alpha} \log \sum_ {m = 1} ^ {M} \left(\frac {| \mathbf {v} [ m ] |}{\sum_ {a = 1} ^ {M} | \mathbf {v} [ a ] |}\right) ^ {\alpha},\tag{18}
$$

where ?? is a column vector of length ??. 

By minimizing the entropy of each column, we will minimize the entropy of the whole matrix, measured by the 2D version: 

$$
R E _ {2 D} (\mathbf {M}) = \frac {1}{1 - \alpha} \log \sum_ {m = 1} ^ {M} \sum_ {n = 1} ^ {L} \left(\frac {| \mathbf {M} [ m , n ] |}{\sum_ {a = 1} ^ {M} \sum_ {b = 1} ^ {L} | \mathbf {M} [ a , b ] |}\right) ^ {\alpha},\tag{19}
$$

where ?? is a matrix of size $M \times L .$ 

In both cases, we typically use $\alpha = 2 .$ . The values of ?? and ?? will be given by the data. 

## 3.3. Optimization

For each column $\widetilde { \mathbf f } _ { n } ,$ we perform an optimization on the matrix $\mathbf { A } _ { n }$ in such a way that ${ \mathbf s } _ { n } = { \mathbf A } _ { n } \tilde { \mathbf f } _ { n }$ corresponds to the highest possible concentration for the reassigned coefficients. This problem is however NP hard, and such an optimization cannot be performed by means of a gradient descent technique. So, we proceed doing coordinate descent [27]. Given an STFT column $\tilde { \mathbf { f } } _ { n }$ to be reassigned, we select the orders, and allocate them in the vector ??, by picking the corresponding $\mathbf { a } _ { m } ^ { n , [ \mathbf { n } [ m ] ] }$ such that the product $\tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ] } \tilde { \mathbf { f } } _ { n }$ has minimum entropy. So we construct our matrix $\tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ] }$ in this way. Once we did this for each ?? in that column $\widetilde { \mathbf f } _ { n } ,$ we call this a cycle. We iterate with more cycles until the matrix changes no more. 

## 3.4. Initialization

We need an initial point for the optimization procedure to start, i.e. a matrix $\tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( 0 ) } ] }$ for a given ??th STFT column. As opposed to [18], where the initial order was fixed for the whole column, here we pursue an initial order that also depends on the frequency. In order to do that, define the support of the column $\widetilde { \mathbf { f } } _ { n }$ (the thresholded STFT) as $\begin{array} { r } { S _ { n } = \bigcup _ { p = 1 } ^ { P _ { n } } s _ { p } ^ { n } ; } \end{array}$ , where the subsets $s _ { p } ^ { n }$ are disjoint. Then, for those ?? ∈ $s _ { p } ^ { n } ,$ we select the order of the initial columns of the matrix such that $R E _ { 1 D } ( | \tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( 0 ) } ] } \tilde { \mathbf { f } } _ { n } \chi [ s _ { p } ^ { n } ] | )$ is minimized. So for each disjoint subset that composes the support of $\tilde { \mathbf { f } } _ { n }$ we simply select a fixed order $\mathbf { n } ^ { ( 0 ) } [ m ] ,$ , for ?? ${ \mathfrak { s e } } ^ { n } { \mathfrak { s } } _ { p } ^ { n } { \mathrm { : } }$ , that minimizes (maximizes) the Rényi entropy (concentration) for those coefficients only, meaning that $\mathbf { n } ^ { ( 0 ) } [ m ]$ is constant for ?? $\in s _ { n } ^ { p } .$ And because of that, we achieve a reassigned column $\tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( 0 ) } ] } \tilde { \mathbf { f } } _ { n }$ that has already a good concentration. 

## 3.5. Coordinate descent

Once one has computed this initial matrix, the optimization pro cedure starts. Since one cannot compute a gradient, one relies on coordinate descent. For that purpose, we define the matrix $\left( \mathbf { A } | \mathbf { a } \right) _ { m } ,$ , in which the ??th column of matrix ?? is replaced by the column vector ??. 

We start our procedure by sorting $| { \tilde { \mathbf { f } } } _ { n } |$ in a decreasing fashion, such that <sup>̃</sup>?? $_ { \iota } [ m _ { 1 } ] | \geq | \tilde { \mathbf { f } } _ { n } [ m _ { 2 } ] | \geq \cdots \geq | \tilde { \mathbf { f } } _ { n } [ m _ { Q _ { n } } ] | ,$ , where $Q _ { n }$ is the number of nonzero coefficients of that STFT column $\tilde { \mathbf { f } } _ { n }$ (i.e. $Q _ { n } = \# S _ { n }$ , the cardinality of the support). Then, for a given ??th iteration the procedure starts defining the new matrix $\tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( i ) } ] ^ { - } } = ( \tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( i - 1 ) } ] } | \mathbf { a } _ { m _ { 1 } } ^ { n , [ \mathbf { n } ^ { ( i ) } [ m _ { 1 } ] ] } ) _ { m _ { 1 } } ^ { \phantom { - } }$ , with $\mathbf { n } ^ { ( i ) } [ m _ { 1 } ] \in$ $\mathcal { N } _ { i }$ , such that $R E _ { 1 D } ( \tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( i ) } ] } \tilde { \mathbf { f } } _ { n } )$ ) is minimal. This simply consists of selecting for that coefficient $\tilde { \bf f } _ { n } [ m _ { 1 } ]$ the order that minimizes (maximizes) the Rényi entropy (concentration). The procedure continues refining the matrix as $\begin{array} { r l r } { \bar { \bf A } _ { n } ^ { [ { \bf n } ^ { ( i ) } ] } } & {  } & { ( \tilde { \bf A } _ { n } ^ { [ { \bf n } ^ { ( i ) } ] } | { \bf a } _ { m _ { ? } } ^ { n , [ { \bf n } ^ { ( i ) } [ m _ { 2 } ] \bar { \bf l } } ) _ { m _ { ? } } } \end{array}$ such that $R E _ { 1 D } ( \tilde { \mathbf { A } } _ { n } ^ { [ \mathbf { n } ^ { ( i ) } ] } \tilde { \mathbf { f } } _ { n } )$ is minimal again, now with the optimal order selected for the coefficient $\tilde { \bf f } _ { n } [ m _ { 2 } ]$ . Then, the procedure continues with the remaining coefficients (positions $m _ { q } , \mathrm { f o r } q = 3 , \ldots , Q _ { n } ) ;$ , always keeping the modification done for the previous ??, to complete a cycle. The next cycle starts again with $\tilde { \mathbf { f } } _ { n } [ m _ { 1 } ]$ to perform the next iteration, and continues with the remaining coefficients. So the idea is to optimize for each coefficient, and once this is done with all the coefficients, one starts a new cycle. The procedure typically converges after a few cycles (typically 3 or 4) to a local minimum (which retrieves however a good solution). Since the Rényi entropy decreases, and as it is always positive, the convergence implies the Rényi entropy varies no more. Note that the possibility of running several cycles of the optimization procedure is another difference with the approach from [18]. 

## 3.6. Algorithm summary

We can summarize the algorithm as follows.

Algorithm 1 Adaptive Synchrosqueezing.

1: Input: $\tilde{\mathbf{F}}_{\tilde{x}}^{g}$ , $\tilde{\omega}_{\tilde{x}}^{[N]}$ for $N \in \mathcal{N} = \{1, \ldots, N_{max}\}$ .

2: for $n = 0, \ldots, L - 1$ do

3: Define the support $S_n = \bigcup_{p=1}^{P_n} s_p^n$ of $\tilde{\mathbf{f}}_n$ .

4: for $p = 1, \ldots, P_n$ do

5: Set $\mathbf{n}^{(0)}[m]$ , for those $m \in s_p^n$ , as $N^*$ , where $N^*$ minimizes $RE_{1D}(\tilde{\mathbf{A}}_n^{[\mathbf{n}^{(0)}}] \tilde{\mathbf{f}}_n \chi[s_p^n])$ .

6: end for

7: Sort $|\tilde{\mathbf{f}}_n|$ in a decreasing fashion ( $|\tilde{\mathbf{f}}_n[m_1]| > |\tilde{\mathbf{f}}_n[m_2]| > \cdots > |\tilde{\mathbf{f}}_n[m_{Q_n}]|$ ).

8: for $i = 1, \ldots, I$ do

9: Assign $\tilde{\mathbf{A}}^{[\mathbf{n}^{(i)}]} = \tilde{\mathbf{A}}^{[\mathbf{n}^{(i-1)}]}$ .

10: for $q = 1, \ldots, Q_n$ do

11: $\tilde{\mathbf{A}}_n^{[\mathbf{n}^{(i)}]} \leftarrow (\tilde{\mathbf{A}}_n^{[\mathbf{n}^{(i)}]} | a_{m_q}^{n,[\mathbf{n}^{*(i)}[m_q]]})_{m_q}$ such that $RE_{1D}(\tilde{\mathbf{A}}_n^{[\mathbf{n}^{(i)}]} \tilde{\mathbf{f}}_n)$ is minimal, where $\mathbf{n}^{*(i)}[m_q]$ is the minimizer order.

12: end for

13: end for

14: Set $\mathbf{s}_n = \tilde{\mathbf{A}}_n^{[\mathbf{n}^{(I)}]} \tilde{\mathbf{f}}_n$ as the $n$ -th column of synchrosqueezed representation $\mathbf{S}_x^g$ .

15: Save $\mathbf{N}[:, n] = \mathbf{n}^{(I)}$ as the optimal orders for the $n$ th STFT column.

16: end for

17: Output: $\mathbf{S}_x^g$ and synchrosqueezing orders for each TF point $[m, n]$ , i.e. the $\mathbf{N}[m, n]$ matrix. 

## 3.7. Computational cost

The computational cost of our adaptive SST would be a function of the number of coefficients to be reassigned. In terms of the notation used so far, this would be $Q _ { n }$ coefficients, for a time instant ??. This number, for a given ??, is usually much smaller than the number of frequency bins ?? since we perform a thresholding (Eq. (4)). 

To compute our adaptive SST, we first need to compute the SSTs up to order $N _ { m a x } = 4 ,$ , which would mean to perform $N _ { m a x } Q _ { n }$ coefficients reallocations. Then, we would need to compute $N _ { m a x }$ Rényi entropies (one for each order). 

Then, we reach the loop for variable ?? (line 8 of Algorithm 1), and there we perform (for each cycle) $N _ { m a x } Q _ { r }$ permutations and computations of Rényi entropy. Since we perform ?? cycles, then the total cost is: $( I + 1 ) N _ { m a x } Q _ { r }$ reallocations, and $( P _ { n } + I Q _ { n } ) N _ { m a x }$ Rényi entropy computations. 

![](images/797aa0b06d23890527feb838b06e3e05968d966549ea5c17e6caac0c5f12053c.jpg)


![](images/0087e4ec00889395bb5d000c825e5a8d1a9da329679b4402b748fad75cf632a9.jpg)



Fig. 2. Results for a multicomponent signal. We show the reassigned TFR with SST2, SST3, SST4, and our proposal SSTa. We zoom in for three different locations. SNR of 30 dB.


![](images/8c68b7a7506e3ac62a9f2b4d704f5551a2774d94f785a0f32724961a0ff948ec.jpg)



Fig. 3. Amplitude estimations for the different studied methods. Portion of the linear chirp of the signal shown in Fig. 2 (green box)


This cost, higher than the original SSTs can be. however, reduced Indeed, the cycle for variable ?? (line 10 in Algorithm 1) can be par allelizable, although we do not explore this in the present paper, and left this topic for future works. Furthermore, as optimization is carried out for each time instants independently, parallelization can also be implemented by optimizing for different time instants simultaneously. 

In the following section, we offer a scatter plot with computational times and number of coefficients, where it can be confirmed that the cost is proportional to $Q _ { n }$ 

## 4. Experiments and results

In this section, we illustrate the capabilities of our proposal. We will denote our proposal as ‘SSTa’, where ‘a’ stands for adaptive. 

## 4.1. Artificial signals

As a first experiment, we show here some results on a multicomponent AM–FM signal. The modulus of the STFT of the signal is depicted on Fig. 2 (left). It consists of a superimposition of a linear chirp and a sinusoidal chirp, both with constant amplitude equal to 1. On Fig. 2 (right) we display both the STFT modulus and the reassigned transforms (SST2, SST3, SST4 and SSTa) for a noisy realization at 30 dB of SNR. More precisely, we zoom in three portions of the TFR of special interest: a part of the linear chirp, of the sinusoidal chirp with strong frequency modulation and of the sinusoidal chirp with strong chirp-rate variations (where the IF attains a local maximum). 

A first observation we make is on the linear chirp and the linear part of the sinusoidal chirp for which we notice that SSTa achieves a sharper representation than the other tested SSTs. Regarding the cyan box, where the sinusoidal chirp achieves a local maximum for the IF, SST2 exhibits some blurring, while SST4 shows ‘hairs’ (those branching structures that seems to emerge from the main ridge). These hairs are in fact the evidence of energy leakage from the mode, with more hair meaning more leakage and therefore less concentration. SSTa, on the other hand, presents almost no blurring and no hairs. 

The quality of the reassignment process can be measured by the IF estimation bias and also by the quality of amplitude estimation. Regarding IF estimation bias, we will present some quantitative results showing errors with respect to the ideal TFR (measured by the Earth’s mover distance, also called optimal transport distance [10,28,29]) in Fig. 4. 

Furthermore, as far as amplitude estimation is concerned, SSTa achieves a much more accurate estimation than the other tested SSTs Note that if the reassignment process is accurate, then the amplitude can be estimated by considering the modulus of the TFR on the ridge, corresponding to a local maximum of that TFR along the frequency axis, and referred to as ??[??] at time ?? in what follows. On Fig. 3, the modulus of the TFR on the ridge is displayed (green box, linear chirp). It can be appreciated that the results provided by SSTa are the best, indicating a superior estimation of amplitude. Estimating the amplitude correctly is important for tasks such as mode reconstruction, for instance for those methods based on local linear chirp approximations [17,30]. 

![](images/39d8aa5e6488e1389d0477bdd62a2212e2df7c6820983925202c35a210a9a11f.jpg)


![](images/eab1dbcb6a55a08a8979027b043f7e6bca1b951e7262147484ec0fc6598db6ed.jpg)


![](images/6c3e086e1ae26f8b95c51f9bcf4341b5f295e573146d3030349840012f72b4ac.jpg)


![](images/fcc498915320da314c0cf687148cf2081dadae5d5d16b46036cffd68397bcd9b.jpg)



Fig. 4. Automatically selected orders for our SSTa method and performances of the studied methods. The selected orders are for SNR of 30 dB, while the entropies and errors explore three different SNRs.


![](images/fca150fee9d92ceb54222f91b72f22b834a8643c25d920f60322145b2d138115.jpg)



Fig. 5. Results for three columns of the STFT (SNR of 20 dB). First row: STFT. Second row: STFT column. Third row: entropies for SSTa as a function of the iteration number. Fourth row: column of SSTa along with the rounded ideal TFR (‘GT’ stands for $[ \phi _ { r e f } ^ { \prime } ( t _ { 0 } ) ]$ , where $t _ { 0 }$ is the corresponding time for each column).


More results are available on Fig. 4, where we show on its left panel, the automatically selected order with SSTa at each TF point. Zooming in the same three areas as in Fig. 2, we see that SSTa locally adapts the order of the synchrosqueezing transform, selecting the order that minimizes (maximizes) the Rényi entropy (concentration). We observe that the orders present a strip arrangement. with a prevalent order for a given fixed time and component. However, SSTa refines the order, especially at those TF points far from the ridge, where the estimations are typically worse with a fixed order. We can also appreciate average results (30 noisy realizations at three different SNRs) for Rényi entropy and error with respect to the ideal TFR. In all the studied cases, the results of our proposal are clearly superior to those of the other methods, achieving more concentrated TFRs, and closer to the ideal ones. It must be mentioned that, although still the best one, the performance of SSTa approaches to that of SST2 for high levels of noise. Thus, to consider adaptive SST in a very noisy context seems not to be beneficial. 

Fig. 5 illustrates the iterative procedure of our proposal. Three columns of the STFT, and the reassigned versions, are shown. The three time instants are marked on the first row (dashed vertical line), and the three columns of the STFT are shown on the second row (we show the modulus of the STFT). The third row shows the Rényi entropy for SSTa, as a function of the iterations (iteration 0 meaning the initial state). We can see how this entropy decreases with the iterations, and how it stagnates after a few cycles. The fourth row shows the results of our SSTa method, along with the rounded ideal instantaneous frequencies. It can be observed how SSTa achieves a concentrated representation, and where some bias appears, it is of at most one frequency bin. 

Regarding computational cost, Fig. 6 presents results for the SSTa at the three time instants put forward in Fig. 5 (30 realizations). The results confirm that the cost is proportional to the number of coefficients $Q _ { n }$ to be reassigned, more so with a correlation coefficient value of $r \ = \ 0 . 9 4 .$ Indeed, out of the three time instants, the one involving the smallest number of coefficients is the middle one, and the left one is associated the highest number of coefficients. This can be appreciated by looking at the second row of Fig. 5. 

![](images/b100ba41d54214f552d1b286a1c95bbc028fb4f93a468e4295933ccf254a47e0.jpg)



Fig. 6. Computational times and number of coefficients (?? ) for the time instants of Fig. 5 (30 realizations).


## 4.2. Real signals

Here we present some examples on real signals. For the computation of Rényi entropy we use $\alpha = 2 ,$ and we indicate the size ?? × ?? of the TFRs. 

## 4.2.1. Bat social vocalization

As a first real example, we show the results on a social vocalization of an Eptesicus fuscus bat [31]. This specific chevron shape call presents some modulation reminiscent to that of sinusoidal chirps. Fig. 7 presents the STFT (with $L = 8 1 9 2$ and ?? = 1024), along with two zoomed in areas where the results of SST2 and SSTa are shown. The Rényi entropies of the STFT, SSTN $( N = 1 , \ldots , 4 )$ , and SSTa are presented, and it can be seen that after SSTa, SST2 is the one with the highest concentration. The ridges of SST2 however are more blurry than those of SSTa, and they present more spurious oscillations. This is confirmed in the plots shown at the bottom row of the figure. Indeed, when evaluating the modulus of the TFR on the ridge, SSTa presents much fewer oscillations than SST2, evidencing a more concentrated TFR with less leakage. 

## 4.2.2. Gibbon song

Gibbons are apes in the family of Hylobatiade, with outstanding vocal displays. As a second real example, we analyze a portion of a gibbon song recorded by the Gibbon Rehabilitation Project.<sup>2</sup>[32] The modulus of the STFT (with ?? = 16384 and ?? = 1024), which presents a distinctive TF signature, can be appreciated in Fig. 8. As with the previous example, we zoom in two areas of the TF plane, showing both SST2 and SSTa. Our SSTa proposal evidences more defined and less blurry ridges, that offer a less oscillating amplitude. SSTa presents the most concentrated TFR, followed by SST2. 

## 4.2.3. Voice signal

As a third real example, we present the analysis of a voice signal from the Saarbruecken Voice Database [33]. This low–high–low type signal is characterized by a sudden change in the pitch, producing an abrupt change in the frequency and therefore a strong frequency modulation. Fig. 9 presents the modulus of the STFT (with ?? = 16384 and $M \ = \ 5 1 2 )$ , along with the results for SST2 and SSTa for two zoomed in areas. SSTa presents again the highest TF concentration, with less blurry ridges which offer better amplitude estimations (much less oscillatory). 

## 4.2.4. Electrocardiogram signal

As a fourth real example, we analyze an electrocardiogram signal from the MIT-BIH Normal Sinus Rhythm Database [34]. Fig. 10 presents the modulus of the STFT (with ?? = 1024 and $M = 5 1 2 )$ , along with the results for SST1 and SSTa for two zoomed-in areas. Several harmonics can be appreciated for this biomedical signal. The Rényi entropy results show that this time SST1 is the second most concentrated representation, after our SSTa proposal. As with the previous cases, SSTa offers a better representation, where the ridges retrieve better amplitude estimations (with fewer oscillations). In this case, we show the instantaneous amplitudes of the middle ridge of each box. 

## 5. Conclusions

In this paper we explored the problem of choosing an optimal order for the SST. We proposed a solution by minimizing the Rényi entropy of the columns of the reassigned representation. We solved this NP hard problem by coordinate descent, efficiently implementing it with sparse matrices. 

The results on simulated and real signals show improvements with respect to fixed order SST. This improvements can be measured both in concentration and in error with respect to ideal TFR. The representations achieved with our SSTa proposal are more defined, less blurry, and they offer better amplitude estimations. 

Limitations arise when the noise content of the signal is high. Our proposal naturally inherits some of the existing limitations of SSTN. Indeed, under a presence of strong noise the IF estimations given by Eq. (8) worsen as ?? increases since more STFTs are needed. Therefore, using adaptive SST in a very noisy context must be done carefully. 

Future works will be devoted to parallel implementations in order to decrease even more the computational cost. In another direction, it would be interesting to investigate signal reconstruction based on SSTa, since we remarked that it offers a smoother estimation of the amplitude on the ridges than original SSTs. 

## CRediT authorship contribution statement

Marcelo A. Colominas: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Sylvain Meignen: Writing – review & editing, Writing – original draft, Visualization, Validation, Supervision, Methodology, Investigation, Funding acquisition, Formal analysis, Data curation, Conceptualization. 

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

## Data availability

Data will be made available on request. 

![](images/b1fcded8fd52823e8df1b648761c38fbc156c00a108eb6860cd68a1e2ae21267.jpg)


![](images/14dde48ec4a5b287b1863db15e9f513b0a63bb512bc63d4cef0829e55bc658f8.jpg)


![](images/24f2d9fa1c2d62fcb0f99a4d363897a36dd05ad5985482e4eaedd0189baef9f7.jpg)


![](images/384130b4ed1b2af9a873d33dce528d46ba2f0f3d48c2db408cde125d0afad31b.jpg)


![](images/9a63e8a66b341ac17f9000736c4f9db1efd2f13150696c0e09d416c34d40a978.jpg)


![](images/6c6d0a6070d4c425606bd061919dca338e9a2b1b904fd06ef59f153c61955dd2.jpg)



Fig. 7. Results for a bat social vocalization. We show the modulus of the STFT, along with two zoomed in areas where we show the modulus of SST2 and SSTa. The Rényi entropies are shown, and also the instantaneous amplitude estimations for SST2 and SSTa.


![](images/762f6484d643f742f4db83a3d28a91ab38789c68d06747b97927309edf17da05.jpg)


![](images/9be3056af6734e41929822c44c1befe3f8e413611c65dab07f8e624a1bc165fe.jpg)


![](images/c1448fed3f4371b30ef162d371eb893d33c6a09b9f5a8bf99e797370725614ed.jpg)


![](images/4f7071783986ee2a3432a76250489e7fd5eed7960898f3e1e3e7006a283965a7.jpg)



Fig. 8. Results for a gibbon song. We show the modulus of the STFT, along with two zoomed in areas where we show the modulus of SST2 and SSTa. The Rényi entropies are shown, and also the instantaneous amplitude estimations for SST2 and SSTa.


![](images/7d9baabf55b88a87f4faa5466b8245921afef67e7a9d3b6d2b7b4df42cf5d035.jpg)


![](images/bad8059024a28165b493e828010122b5b6560a0dd3b5c44f5e2805e177cd925c.jpg)



Fig. 9. Results for a voice signal. We show the modulus of the STFT, along with two zoomed in areas where we show the modulus of SST2 and SSTa. The Rényi entropies are shown. and also the instantaneous amplitude estimations for SST2 and SSTa


![](images/6add8674843e76639f841ca00e5d9d3c6c0407515d64042cf405374b36430d76.jpg)



Fig. 10. Results for an electrocardiogram signal. We show the modulus of the STFT, along with two zoomed in areas where we show the modulus of SST1 and SSTa. The Rényi entropies are shown, and also the instantaneous amplitude estimations for SST1 and SSTa (for the middle ridge in each box).


## References



[1] R. Gribonval, E. Bacry, Harmonic decomposition of audio signals with matching pursuit, IEEE Trans. Signal Process. 51 (1) (2003) 101–111. 





[2] C.L. Herry, M. Frasch, A.J. Seely, H.-T. Wu, Heart beat classification from singlelead ECG using the synchrosqueezing transform, Physiol. Meas. 38 (2) (2017) 171–187. 





[3] Y.-Y. Lin, H.-T. Wu, C.-A. Hsu, P.-C. Huang, Y.-H. Huang, Y.-L. Lo, Sleep apnea detection based on thoracic and abdominal movement signals of wearable piezoelectric bands, IEEE J. Biomed. Health Inf. 21 (6) (2017) 1533–1545. 





[4] P. Flandrin, Time-Frequency/Time-Scale Analysis, Academic Press, 1998. 





[5] C.K. Chui, Q. Jiang, L. Li, J. Lu, Analysis of an adaptive short-time Fourier transform-based multicomponent signal separation method derived from linear chirp local approximation, J. Comput. Appl. Math. 396 (2021) 113607. 





[6] I. Daubechies, J. Lu, H.-T. Wu, Synchrosqueezed wavelet transforms: an empirical mode decomposition-like tool, Appl. Comput. Harmon. Anal. 30 (2) (2011) 243–261. 





[7] G. Thakur, H.-T. Wu, Synchrosqueezing-based recovery of instantaneous frequency from nonuniform samples, SIAM J. Math. Anal. 43 (5) (2011) 2078–2095. 





[8] T. Oberlin, S. Meignen, V. Perrier, The Fourier-based synchrosqueezing transform, in: 2014 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2014, pp. 315–319, http://dx.doi.org/10.1109/ICASSP.2014 6853609. 





[9] T. Oberlin, S. Meignen, V. Perrier, Second-order synchrosqueezing transform or invertible reassignment? towards ideal time-frequency representations, IEEE Trans. Signal Process. 63 (5) (2015) 1335–1344, http://dx.doi.org/10.1109/TSP. 2015.2391077 





[10] R. Behera, S. Meignen, T. Oberlin, Theoretical analysis of the second-order synchrosqueezing transform, Appl. Comput. Harmon. Anal. 45 (2) (2018) 379–404. 





[11] D.H. Pham, S. Meignen, High-order synchrosqueezing transform for multicom ponent signals analysis-with an application to gravitational-wave signal, IEEE Trans. Signal Process. 65 (12) (2017) 3168–3178. 





[12] L. Li, H. Cai, Q. Jiang, Adaptive synchrosqueezing transform with a time-varying parameter for non-stationary signal separation, Appl. Comput. Harmon. Anal. (2019). 





[13] L. Li, H. Cai, H. Han, Q. Jiang, H. Ji, Adaptive short-time Fourier transform and synchrosqueezing transform for non-stationary signal separation, Signal Process. 166 (2020) 107231. 





[14] R. Zhang, Z. Wang, Y. Tan, X. Yang, S. Yang, Local maximum frequencychirp-rate synchrosqueezed chirplet transform, Digit. Signal Process. 130 (2022) 103710. 





[15] Z. Chen, H.-T. Wu, Disentangling modes with crossover instantaneous frequen cies by synchrosqueezed chirplet transforms, from theory to application, Appl. Comput. Harmon. Anal. 62 (2023) 84–122. 





[16] X. Zhu, H. Yang, Z. Zhang, J. Gao, N. Liu, Frequency-chirprate reassignment, Digit. Signal Process. 104 (2020) 102783. 





[17] N. Laurent, M.A. Colominas, S. Meignen, On local chirp rate estimation in noisy multicomponent signals: With an application to mode reconstruction, IEEE Trans. Signal Process. 70 (2022) 3429–3440. 





[18] M.A. Colominas, S. Meignen, Making synchrosqueezing locally adaptive in the time-frequency plane, in: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, IEEE, 2023, pp. 1–5. 





[19] D.L. Donoho, I.M. Johnstone, Ideal spatial adaptation by wavelet shrinkage, Biometrika 81 (3) (1994) 425–455. 





[20] S. Mallat, A Wavelet Tour of Signal Processing: The Sparse Way, Academic Press, 2008. 





[21] D.-H. Pham, S. Meignen, A novel thresholding technique for the denoising of multicomponent signals, in: IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, IEEE, 2018, pp. 4004–4008. 





[22] I. Daubechies, S. Maes, A nonlinear squeezing of the continuous wavelet transform based on auditory nerve models, Wavelets Med. Biol. (1996) 527–546. 





[23] H.-T. Wu, Adaptive Analysis of Complex Data Sets (Ph.D. thesis), Princeton, 2011. 





[24] S. Meignen, N. Singh, Analysis of reassignment operators used in synchrosqueezing transforms: With an application to instantaneous frequency estimation, IEEE Trans. Signal Process. 70 (2021) 216–227. 





[25] N. Laurent, S. Meignen, A new adaptive technique for multicomponent signals reassignment based on synchrosqueezing transform, in: 2022 30th European Signal Processing Conference, EUSIPCO, IEEE, 2022, pp. 2136–2140. 





[26] R.G. Baraniuk, P. Flandrin, A.J. Janssen, O.J. Michel, Measuring time-frequency information content using the Rényi entropies, IEEE Trans. Inform. Theory 47 (4) (2001).1391-1409 





[27] S.J. Wright, Coordinate descent algorithms, Math. Program. 151 (1) (2015) 3–34. 





[28] O. Pele, M. Werman, Fast and robust earth movers distances, in: 2009 IEEE 12th International Conference on Computer Vision, IEEE, 2009, http://dx.doi.org/10. 1109/iccv 2009.5459199 





[29] I. Daubechies, Y.G. Wang, H.-T. Wu, ConceFT: concentration of frequency and time via a multitapered synchrosqueezed transform, Phil. Trans. R. Soc. A 374 (2065) (2016) http://dx.doi.org/10.1098/rsta.2015.0193 





[30] N. Laurent, S. Meignen, A novel time-frequency technique for mode retrieval based on linear chirp approximation, IEEE Signal Process. Lett. 27 (2020) 935–939. 





[31l J. Montova. Y. Lee, A. Salles. Social communication in big brown bats. Front Ecol, Evol, 10.(2022).903107 





[32] S. Broad, J. Jenkins, et al., Gibbons in their midst? Conservation volunteers’ motivations at the Gibbon Rehabilitation Project, Phuket, Thailand, J. Discov. Volunt Tourism: Int. Case Study Perspect. (2008) 72–85. 





[33] B. Woldert-Jokisz, Saarbruecken voice database, 2007, URL http://stimmdb.coli. uni-saarland.de 





[34] A.L. Goldberger, L.A. Amaral, L. Glass, J.M. Hausdorff, P.C. Ivanov, R.G. Mark, J.E. Mietus. G.B. Moody. C.-K. Peng. H.E. Stanley. PhysioBank. PhysioToolkit and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 101 (23) (2000) e215–e220. 

