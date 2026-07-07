# Application of High-Order Multisynchrosqueezing Transform in Fault Diagnosis

Wenjie Bao , Zhen Liu , Songyong Liu , and Fucai Li 

Abstract— The time–frequency analysis (TFA) is an important tool to analyze the nonstationary signal in mechanical fault diagnosis. Nevertheless, the classical TFA methods fail to generate a crisper and focused time–frequency representation (TFR) for the strongly time-varying signal in mechanical system. How to improve the readability of TFR is a challenging task. The present research introduces a new TFA method termed high-order multisynchrosqueezing transform (HMST) for dealing with nonstationary signal with strongly time-varying features. First, a signal model for nonstationary signal is constructed. Then, on the basis of high-order Taylor expansions of this signal model, an explicit formula for high-order instantaneous frequency (IF) estimation is drawn. Finally, TF energy is rearranged to this estimated IF from the original position by iterative operations. The validity of the proposed method for enhancing the TF concentration and accuracy is proved by comparing it with some advanced techniques using numerical signals and bearing fault experimental signals. The validations indicate that our proposed method performs much better over other classical TFA methods in bearing fault diagnosis. 

Index Terms— Fault diagnosis, nonstationary signal, synchrosqueezing transform (SST), time–frequency analysis (TFA). 

## I. INTRODUCTION

process nonstationary signals in fault diagnosis [1], [2], [3], [4], [5], [6], [7], because it can observe both temporal and frequency features simultaneously instead of separately [8]. The short-time Fourier transform (STFT) and the continuous wavelet transform (CWT) are the most well-known TFA methods, and they can extend the 1-D time-domain signals to the 2-D TF plane and uncover time-varying features of mechanical system fault signals [9], [10]. Nevertheless, both TFA methods are linear and subject to the Heisenberg uncertainty principle to achieve maximum TF resolution. Then, the quadratic TFA method named Wigner–Ville distribution (WVD) is proposed to improve the TF concentration and applied in wind turbine fault diagnosis [11]. Nevertheless, it inevitably generates the cross-term interference when dealing with multicomponent signals. In order to suppress the cross-term, some efforts have been made [12], but they all eventually lead to the blurry TFR. 

To solve this issue, Michel and Gueguen [13] proposed reassignment method (RM) to increase TF concentration. The RM is the TF postprocessing method, and it can rearrange the coefficients into the ridge curves [14]. Nevertheless, RM fails to reconstruct the signal from TFR. This hinders the actual applications. Generally, invertibility is one of the most significant properties for the TFA method for applications in fault diagnosis [15]. Thus, the STFT-based and wavelet-based synchrosqueezing transform (SST) emerge as the times require and are widely applied in mechanical fault diagnosis, such as gearbox, bearing, aeroengine [16], [17], and so on. Like the RM, the SST also sharpens the TFR through rearranging the coefficients, but it retains the crucial invertibility. Since the readability of SST-based TFR is much superior to traditional TFA methods, the SST-based TFA method enters a rapid development stage. In addition, the second-order SST (SST2) is presented and employed to wind turbine fault diagnosis [18]. Nevertheless, SST and SST2 will no longer be valid for signals with strongly frequency-varying characteristics. 

To tackle this difficulty, the high-order SST (HSST) is derived as a matter of course, and it is applied to deal with extremely oscillatory signals in planetary gearbox fault diagnosis [19]. HSST is capable of dramatically improving the TF concentration by increasing the instantaneous frequency (IF) estimation order. Nevertheless, high-order IF estimation has no explicit formula, and it is derived utilizing recursion. The manual deduction is necessary for each high-order IF estimation, which results in huge computation work. For example, the N th-order HSST requires to execute O(3N − 1) STFT operations and a lot of differentiation operations. It hinders the practical applications of HSST, so no actual applications on fourth-order or higher IF estimation are available. This development bottleneck continued for quite some time until the advent of multisynchrosqueezing transform (MSST) and its application to fault diagnosis of rolling bearings [20]. The MSST can effectively enhance the TF concentration by iterative procedure, and it has a lower computational cost compared with HSST while acquiring the same concentration of the TFR. However, the MSST never altered the fact that it is very complicated to calculate the higher order IF estimation. Instead, it chooses linear IF estimation as the position to rearrange the 

TF energy. Therefore, the MSST fails to generate an accurate TFR when analyzing the nonstationary signals with strongly time-varying features no matter how many iterative operations. For the strongly nonlinear mode in mechanical fault diagnosis, the MSST is incapable of producing a precise TFR, although it can produce a sparse TFR. 

In this article, a kind of generalized SST-based TFA method is presented for processing strongly time-varying signals in bearing fault diagnosis, that is termed high-order multisynchrosqueezing transform (HMST). The HMST is based on HSST, but it rederives high-order IF estimation and gives explicit formulate. It decreases the amount of STFT operations, avoiding a large number of differentiation and division operations at the same time. It means the proposed method can be easily implemented by programming and applied to engineering practice. Next, the algorithm concentrates the TF energy on this generalized IF estimation by iterative reassignment procedure. Since the high-order IF estimation is the unbiased estimation of true IF, the HMST can improve TF accuracy while improving TF concentration. Moreover, the proposed method retains the ability to reconstruct the signal. 

This article is structured as follows. The theoretical basics are given in Section II. The HMST is described in Section III. The numerical validation is shown in Section IV. The experimental validation is shown in Section V. The conclusion is drawn in Section VI. 

## II. THEORETICAL BASICS

## A. STFT and Gaussian Window Function

The STFT of the function $x ( t ) \in L ^ { 2 } ( \mathbb { R } )$ with the window function $g ( t ) \in L ^ { 2 } ( \mathbb { R } )$ is defined as 

$$
V _ {x} ^ {w} (t, \omega) = \int_ {- \infty} ^ {\infty} x (\tau) w ^ {*} (\tau - t) e ^ {- j \omega (\tau - t)} d \tau\tag{1}
$$

where $w ^ { * }$ denotes the complex conjugate of $w .$ 

The STFT can be derived in another form 

$$
V _ {x} ^ {w} (t, \omega) = \int_ {- \infty} ^ {\infty} x (\tau + t) w ^ {*} (\tau) e ^ {- j \omega (\tau)} d \tau .\tag{2}
$$

Thus, the STFT partial derivative concerning t is obtained 

$$
\frac {\partial}{\partial t} V _ {x} ^ {w} (t, \omega) = - V _ {x} ^ {w ^ {\prime}} (t, \omega) + j \omega V _ {x} ^ {w} (t, \omega)\tag{3}
$$

where $V _ { x } ^ { w ^ { \prime } } ( t , \omega )$ means STFT using $w ^ { \prime } ( t ) = d w ( t ) / d t$ as the window function. 

Also, the STFT partial derivative with ω is obtained 

$$
\frac {\partial}{\partial \omega} V _ {x} ^ {w} (t, \omega) = - j V _ {x} ^ {t w} (t, \omega).\tag{4}
$$

The Gaussian window function w(t) with unit norm is defined as 

$$
w (t) = p \cdot e ^ {\frac {q}{2} t ^ {2}}\tag{5}
$$

where $p = ( \pi \sigma ^ { 2 } ) ^ { - ( 1 / 4 ) }$ and $q = - ( 1 / \sigma ^ { 2 } )$ are the constants. Without loss of generality, all the Gaussian window functions are represented by w in this article. 

## B. Multisynchrosqueezing Transform

The SST can be expressed as 

$$
\mathrm{SST} (t, \omega) = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \big (\omega - \hat {\omega} (t, \eta) \big) d \eta\tag{6}
$$

where $\delta ( * )$ is the Dirac delta function, and $\hat { \omega } ( t , \eta )$ means 2-D IF estimation. 

The MSST is practical and innovative method proposed recently in [20] to improve the concentration of the TFR. It can be defined as 

$$
\begin{array}{c} \mathrm{MSST} ^ {[ 2 ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{SST} (t, \eta) \delta \big (\omega - \hat {\omega} (t, \eta) \big) d \eta \\ \mathrm{MSST} ^ {[ 3 ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{MSST} ^ {[ 2 ]} (t, \eta) \delta \big (\omega - \hat {\omega} (t, \eta) \big) d \eta \\ \vdots \\ \mathrm{MSST} ^ {[ M - 1 ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{MSST} ^ {[ M - 2 ]} (t, \eta) \delta \big (\omega - \hat {\omega} (t, \eta) \big) d \eta \\ \mathrm{MSST} ^ {[ M ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{MSST} ^ {[ M - 1 ]} (t, \eta) \delta \big (\omega - \hat {\omega} (t, \eta) \big) d \eta \end{array}\tag{7}
$$

where $\mathbf { M S S T } ^ { [ M ] }$ is the result of the STFT-based TFR dealt with squeezing transforms for M times. 

## III. HIGH-ORDER MSST

## A. Signal Model and Its Properties

The single component signal is assumed to be estimated from a frequency and amplitude modulated signal 

$$
x (\tau) = A (\tau) e ^ {j \varphi (\tau)}\tag{8}
$$

where $A ( \tau )$ is the amplitude-modulated function and $\varphi ( \tau )$ is the frequency-modulated function. Then, we can obtain its N th-order Taylor expansion for τ near t as 

$$
x (\tau) = \exp \left(\sum_ {k = 0} ^ {N} \frac {\log [ A ^ {(k)} (t) ] + j \varphi^ {(k)} (t)}{k !} (\tau - t) ^ {k}\right)\tag{9}
$$

where $Z ^ { ( k ) } ( t )$ is the kth derivative of $Z ( t )$ 

Furthermore, we can deduce the derivative of this signal model as follows: 

$$
\frac {d x (\tau)}{d \tau} = x (\tau) \cdot \left(\sum_ {k = 1} ^ {N} \frac {\log [ A ^ {(k)} (t) ] + j \varphi^ {(k)} (t)}{(k - 1) !} (\tau - t) ^ {k - 1}\right).\tag{10}
$$

Combining (2), the partial derivative of the STFT of (10) holds as 

$$
\frac {\partial}{\partial t} V _ {x} ^ {w} (t, \omega) = \sum_ {k = 1} ^ {N} Q _ {k} (t) V _ {x} ^ {t ^ {k - 1} w} (t, \omega)\tag{11}
$$

where $Q _ { k } ( t ) ~ = ~ ( ( \log [ A ^ { ( k ) } ( t ) ] + j \varphi ^ { ( k ) } ( t ) ) / ( ( k - 1 ) ! ) )$ , and $V _ { x } ^ { t ^ { k - 1 } w } ( t , \omega )$ is the STFT result with the analysis window $t ^ { \check { k } - 1 } w ( t )$ 

## B. High-Order IF Estimation

At the point (t, ω), the high-order IF estimation $\hat { \omega } _ { [ N ] } ( t , \omega )$ of the signal model is defined as 

$$
\hat {\omega} _ {[ N ]} (t, \omega) = \varphi^ {\prime} (t) = \Im (Q _ {1} (t))\tag{12}
$$

where I(·) denotes the imaginary part. Thus, the calculation of IF estimation is converted into solving $Q _ { 1 } ( t )$ 

Expanding (11) and dividing by $V _ { x } ^ { w } ( t , \omega )$ $Q _ { 1 }$ can be expressed as 

$$
\eta_ {x} (t, \omega) = Q _ {1} + \sum_ {k = 2} ^ {n} Q _ {k} (t) \frac {V _ {x} ^ {t ^ {k - 1} w} (t , \omega)}{V _ {x} ^ {w} (t , \omega)}\tag{13}
$$

where $\eta _ { x } ( t , \omega ) = [ \partial _ { t } V _ { x } ^ { g } ( t , \omega ) ] / V _ { x } ^ { g } ( t , \omega )$ 

Rewriting (13) as the form of matrix 

$$
b _ {1} (t, \omega) = \mathbf {A} _ {n} (t, \omega) \mathbf {Q} _ {n} (t) ^ {\mathrm{T}}\tag{14}
$$

with 

$$
\begin{array}{l} b _ {1} (t, \omega) = \eta_ {x} (t, \omega) \\ \mathbf {A} _ {n} (t, \omega) = \left[ \begin{array}{c c c} 1 & a _ {2, 1} (t, \omega) & a _ {3, 1} (t, \omega), \ldots , a _ {n, 1} (t, \omega) \end{array} \right] \\ \mathbf {Q} _ {n} (t) = \left[ \begin{array}{c c c} Q _ {1} (t) & Q _ {2} (t) & Q _ {3} (t), \ldots , Q _ {n - 1} (t) & Q _ {n} (t) \end{array} \right] \\ a _ {k, 1} (t, \omega) = \frac {V _ {x} ^ {t ^ {k - 1} w} (t , \omega)}{V _ {x} ^ {w} (t , \omega)} \end{array}
$$

where $( \cdot ) ^ { \mathrm { T } }$ means the matrix transpose. 

Comparing (3) and (4), we can find that the derivative of STFT concerning with ω results in simper expressions. Thus, using the derivative of (14) concerningω, we can obtain the second equation as 

$$
\begin{array}{c c c} \partial_ {f} b _ {1} (t, \omega) = [ 0 & \partial_ {\omega} a _ {2, 1} (t, \omega) & \partial_ {\omega} a _ {3, 1} (t, \omega) \quad \dots \\ & & \partial_ {\omega} a _ {n, 1} (t, \omega) ] \mathbf {Q} _ {n} (t) ^ {\mathrm{T}}. \end{array}\tag{15}
$$

Similarly, we can obtain the third equation by using the derivative of (15) concerning ω. And so on, we can get an equation with N -order matrix. Then, the $N \times N$ coefficient matrix can be converted into an upper triangular matrix 

$$
\left[ \begin{array}{c} b _ {1} \\ b _ {2} \\ \vdots \\ b _ {N - 1} \\ b _ {N} \end{array} \right] = \left[ \begin{array}{c c c c c} 1 & a _ {2, 1} & a _ {3, 1} & \dots & a _ {N, 1} \\ 0 & 1 & a _ {3, 2} & \dots & a _ {N, 2} \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & \dots & a _ {N, N - 1} \\ 0 & 0 & 0 & \dots & 1 \end{array} \right] \left[ \begin{array}{c} Q _ {1} \\ Q _ {2} \\ \vdots \\ Q _ {N - 1} \\ Q _ {N} \end{array} \right]\tag{16}
$$

with 

$$
\begin{array}{l} b _ {k} = \left\{ \begin{array}{l l} \eta_ {x} (t, \omega), & k = 1 \\ \frac {\partial_ {\omega} b _ {k - 1}}{\partial_ {\omega} a _ {k , k - 1}}, & k = 2, \ldots , N \end{array} \right. \\ a _ {i, j} = \left\{ \begin{array}{l l} \frac {V _ {x} ^ {t ^ {i - 1} g} (t , \omega)}{V _ {x} ^ {g} (t , \omega)}, & i = 1, 2, \ldots , n; j = 1 \\ \frac {\partial_ {\omega} a _ {i , j - 1}}{\partial_ {\omega} a _ {j , j - 1}}, & i = j, \ldots , n; j = 2, \ldots , n. \end{array} \right. \end{array}
$$

By solving the above equation system in $( 1 6 ) , \ Q _ { 1 }$ can be obtained 

$$
Q _ {1} (t) = \eta_ {x} (t, \omega) - \sum_ {k = 2} ^ {n} a _ {k, 1} (t, \omega) P _ {k} (t, \omega)\tag{17}
$$

with 

$$
P _ {k} (t, \omega) = \left\{ \begin{array}{l l} b _ {N}, & k = N \\ b _ {k} (t, \omega) - \sum_ {i = k + 1} ^ {N} a _ {i, k} (t, \omega) P _ {i} (t, \omega) \\ & k = N - 1, \ldots , 2. \end{array} \right.
$$

Thus, the N -order IF estimation can be derived as 

$$
\hat {\omega} _ {[ N ]} (t, \omega) = \left\{ \begin{array}{l l} 2 \pi \omega + j \Im \bigg (\frac {V _ {x} ^ {w ^ {\prime}} (t , \omega)}{V _ {x} ^ {w} (t , \omega)} \bigg), & N = 1 \\ \Im (Q _ {1} (t)), & N \geq 2. \end{array} \right.\tag{18}
$$

## C. High-Order Multisynchrosqueezing

Combining (7) and (18), the expression of the HMST can be defined as 

$$
\begin{array}{r l} & {\mathrm{HMST} ^ {[ 1, N ]} (t, \omega) = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta (\omega - \hat {\omega} _ {[ N ]} (t, \eta)) d \eta} \\ & {\mathrm{HMST} ^ {[ 2, N ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{HMST} ^ {[ 1, N ]} (t, \eta) \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \eta) \big) d \eta} \end{array}
$$

$$
\begin{array}{c} \mathrm{HMST} ^ {[ M, N ]} (t, \omega) = \int_ {- \infty} ^ {\infty} \mathrm{HMST} ^ {[ M - 1, N ]} (t, \eta) \\ \times \delta (\omega - \hat {\omega} _ {[ N ]} (t, \eta)) d \eta . \end{array}\tag{19}
$$

If we use $\hat { \omega } _ { [ N ] } ^ { [ M ] } ( t , \omega )$ to denote the IF estimation of HMST, the expression of the HMST can also be defined as 

$$
\mathrm{HMST} ^ {[ M, N ]} (t, \omega) = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\omega - \hat {\omega} _ {[ N ]} ^ {[ M ]} (t, \eta) \Big) d \eta\tag{20}
$$

with 

$$
\hat {\omega} _ {[ N ]} ^ {[ M ]} (t, \eta) = \underbrace {\hat {\omega} _ {[ N ]} \big (t , \hat {\omega} _ {[ N ]} \big (t , \ldots , \hat {\omega} _ {[ N ]} \big (t , \ldots , \hat {\omega} _ {[ N ]} (t , \eta) \big) \cdots \big) \big)} _ {\text {Iterating for M times}}
$$

where $\hat { \omega } _ { [ N ] } ( t , \eta )$ can be obtained in (18). Equation (20) is proved in Appendix. 

In theory, the signal can be reconstructed by integrating the HMST result. The HMST reconstruction formular can be obtained as 

$$
x (t) = \frac {1}{2 \pi w ^ {*} (0)} \int_ {- \infty} ^ {\infty} \mathrm{HMST} ^ {[ M, N ]} (t, \omega) d \omega .\tag{21}
$$

So, we can obtain the reconstructed signal 

$$
x (t) = \frac {1}{2 \pi w ^ {*} (0)} \int_ {- \infty} ^ {\infty} \mathrm{HMST} ^ {[ M, N ]} (t, \omega) d \omega .
$$

## IV. NUMERICAL VALIDATION

In this section, we demonstrate the superiorities of the proposed method in analyzing the signal with strongly IF through the numerical signal. The main focus of performance is TF accuracy and TF concentration. The constructed multicomponent simulation signal contains two components, one containing strong time-varying IF and one containing linear IF, and the simulation signal is defined as 


Time(s)


![](images/11c4f59fa324e76d25b1edcc14bf94f9d948683919378e757d51f60c388e337b.jpg)


![](images/e63e318d1481c98234a460b3e1fe417af58ba023dc2e7459a945338de617d245.jpg)



Fig. 1. (a) Time waveform of the original signal and (b) STFT result.


![](images/ce2b36f3411b8fa8ac33e7e62305ba846b4b3ba6ee24040c58cb2d5199f5d748.jpg)



(a)


![](images/5545270e4b9872f1782ba5264004f3b4f4194194ff69367ae1b7b13ab7d76657.jpg)



(b)


![](images/779c7d4cc3d6f5527a86fd8c6781db260056113cfd721bfd3dafb7ef82f7f924.jpg)


![](images/d4a2d20bed98f612baa28d7c39cb15999bb1043852c1fecdd6f04166a196bdc9.jpg)



(c)



(d)


![](images/74cf307707cb307ced40bac7006f565535972a2870edf81fe11f55ba9a4f6fcd.jpg)



(e)


![](images/48c15ade3e8270b6acd397ed4a4f7388051e382c0ec94703dae15ff7af3017df.jpg)



Fig. 2. TFRs obtained by (a) STFT, (b) SST, (c) SET, (d) MSST, (e) HSST, and (f) HMST.


$$
x (t) = x _ {1} (t) + x _ {2} (t)\tag{22}
$$

with 

$$
\left\{ \begin{array}{l} x _ {1} (t) = \sin (2 \pi \cdot (3 0 0 t - 1. 5 \cdot \cos (1 4 \pi t))) \\ x _ {2} (t) = \sin \bigl (2 \pi \cdot \bigl (7 5 t + 2 5 t ^ {2} \bigr) \bigr) \end{array} \right.\tag{23}
$$

where the sampling frequency is 1000 Hz. Fig. 1 presents the original signal time waveform and its true IFs. 

First, we obtain the TFR results generated from various TFA methods, as illustrated in Fig. 2. It can be found that the TFR results of the STFT, the SST, and the HSST are dispersed. In comparison, the TFR results of the SET, the MSST, and the proposed HMST have better performance in TF concentration. The respective enlarged plots of different TF postprocessing methods are illustrated in Fig. 3 and the red lines are the true IFs. By comparing these TFR results, we explore the performance of the HMST in TF accuracy and TF concentration. As a classical TF postprocessing method, the SST has a good performance for component $x _ { 2 } ( t ) { \ ; }$ nevertheless, it cannot produce an accurate and concentrated TFR for component $x _ { 1 } ( t )$ in Fig. 3(a). Compared with the SST, the SET has more concentrated TFR result, but it also fails to have an accurate performance for component $x _ { 1 } ( t )$ in Fig. 3(b). In Fig. 3(c), although the MSST can produce a concentrated TFR by multiple squeezing transformations, it is still incapable of generating accurate TFR for component $x _ { 1 } ( t )$ because it uses the same low-order IF estimation as the previous two TFA methods. As shown in Fig. 3(d), the HSST can produce an accurate TFR for $x _ { 1 } ( t )$ due to its higher order IF estimation, but its TFR is less concentrated than that of the MSST. Thus, the above methods cannot generate a good TFR result for the component with strongly time-varying IF. However, HMST can produce the best TFR for both the components in Fig. 3(e), because it combines the advantages of both the MSST and the HSST. 

![](images/fc01748e0ba3e0803e39cdd7170df506574a465b87e76bbe736828fd361bcb80.jpg)


![](images/9f2e473ecbb7ae73b4914257fb4c71b54863a91e66f9ae8b75c41f4dfc12992d.jpg)


![](images/72a10e0f6caac6b330c787cb5b79c23a318f279463e1760d0dbae9226125cb91.jpg)


![](images/b2ceb55f13b28bc9ff6109a06641200926b05c48d9d0e03ea461b71b0ffa7a0f.jpg)



(b)


![](images/19b28fa9a2ba824f2ac5e54b555e0dfab5985989a0ab5d1a03adb5c8fac0777e.jpg)


![](images/67a9f6c80c676fab4ac3739494842fd707e284dab9a39c0cc75ea232e05ed7e0.jpg)



(c)


![](images/2a9d71c48dcf2d7a1c8cc02962d179e3fa7a797e54b001d76ec6384fceb3887d.jpg)


![](images/77f019d56b5da8851c12ae86bb52e0a179c9ad2822f72e685b6ace48b1904ed5.jpg)


![](images/430a2384805094b0d31d3314cba8bcf70dadac4f2d2f3609dd08bb3d07b6eed4.jpg)


![](images/cfdb572a344a76dcd1ccde299c8bd474490e34bcc49b64596d59eadb1500125c.jpg)



Fig. 3. Enlarged TFRs of (a) SST, (b) SET, (c) MSST, (d) HSST, and (e) HMST.



TABLE I



EMD VALUES FOR VARIOUS TFA METHODS


<table><tr><td>Method</td><td>STFT</td><td>SST</td><td>SET</td><td>MSST</td><td>HSST</td><td>HMST</td></tr><tr><td><eq>EMD (x_1)</eq></td><td>2.858</td><td>2.792</td><td>2.494</td><td>2.514</td><td>1.056</td><td>0.431</td></tr><tr><td><eq>EMD (x_2)</eq></td><td>2.715</td><td>0.880</td><td>0.878</td><td>0.866</td><td>0.879</td><td>0.863</td></tr></table>


TABLE II



RE VALUES FOR VARIOUS TFA METHODS


<table><tr><td>Method</td><td>STFT</td><td>SST</td><td>SET</td><td>MSST</td><td>HSST</td><td>HMST</td></tr><tr><td>RE (<eq>x_{1}</eq>)</td><td>6.432</td><td>4.682</td><td>2.342</td><td>1.099</td><td>1.436</td><td>0.039</td></tr><tr><td>RE (<eq>x_{2}</eq>)</td><td>5.392</td><td>0.136</td><td>0.224</td><td>0.068</td><td>0.106</td><td>0.025</td></tr></table>

Furtherly, we also use three objective quantified indicators to evaluate TF accuracy and TF concentration for the TFA methods. The first indicator is the Earth mover’s distance (EMD), and it is used to evaluate the accuracy of TFR by calculating standard deviation between ideal IF and TFR result [21]. The smaller EMD means the more accurate TFR. The EMD is defined as 

$$
\mathrm{EMD} = \iint_ {s} | V _ {n} - V _ {i} | d s\tag{24}
$$

where $V _ { n }$ is the calculated TFR result and $V _ { i }$ is the ideal TFR. It can be seen in Table I that the TFR generated by HMST owns the smallest EMD, which means the proposed method has the best TF accuracy. For component $x _ { 2 } ,$ the TF accuracy of other methods is almost the same except for the STFT, because the component $x _ { 2 }$ has a fixed frequency. For component $x _ { 1 }$ , the HSST and the proposed HMST have much better performance in TF accuracy than other methods, because the IF of component $x _ { 1 }$ is more complex and requires high-order IF to estimate. Therefore, the proposed HMST can effectively improve the TF accuracy for the signal component with complex IF. 

The second indicator is the Rényi entropy (RE), and it is usually regarded as a quantitative indicator measuring TF concentration [15]. The smaller RE represents the more concentrated TFR. The RE is defined as 

$$
R = - \frac {1}{2} \log_ {2} \frac {\iint_ {R} ^ {2} | V (t , \omega) | ^ {3} d \omega d t}{\iint_ {R} ^ {2} | V (t , \omega) | d \omega d t}\tag{25}
$$

where $V ( t , \omega )$ means analyzed TFR. It can be found in Table II that the TFR obtained by HMST has the smallest RE value, and this implies that HMST provides the best performance in TF concentration. For component $x _ { 2 } ,$ increasing the order of the IF estimation does not have much effect on improving the TF concentration of the SST-based methods, because the component $x _ { 2 }$ has a fixed frequency. For component $x _ { 1 } ,$ increasing the order of the IF estimation and the number of the SST can significantly improve the TF concentration of the TFR, because the IF of component $x _ { 1 }$ is more complex and requires high-order IF to estimate. Therefore, the proposed HMST can effectively improve the TF concentration for the signal component with complex IF. 

We also use another quantified indicator, normalized energy (NE) [22], to evaluate TF concentration. The NE first normalized the total energy of TFR, then arranges the energy points from large to small, and stacks them one by one. The NE value of the TFR is close to 1 with fewer stacked energy points, indicating that the TFR is more concentrated. Thus, we give the NE values of TFRs of component $x _ { 1 } ( t )$ obtained by different TFA methods versus number of coefficients in Fig. 4. The NE value for the STFT increases slightly and that of SST and SET increases well. This shows that the TFRs of the STFT, SST, and SET are not concentrated enough. Compared with the above three methods, the MSST and the HSST have relatively better performance in TF concentration, and the MSST is slightly better than the HSST. The NE of proposed HMST increases most rapidly, approaching 1 at approximately N TF points. This implies that the energy of the TFR concentrates mainly on the TFR ridge. Therefore, the proposed HMST has the best performance in TF concentration. 

![](images/03916e0a384ad93b3f2233df3e4d8544d8ea42ff130182b945750d7cb433318f.jpg)



Fig. 4. NE values for various TFA methods versus the number of coefficients.



TABLE III



BEARING PARAMETERS


<table><tr><td>Bearing type</td><td>Pitch diameter (D)</td><td>Ball diameter (d)</td><td>Number of balls (n)</td><td>Contact angle (β)</td></tr><tr><td>ER16K</td><td>35.52 mm</td><td>7.94 mm</td><td>9</td><td>0 deg</td></tr></table>

In conclusion, the results of quantitative analysis of TFA methods by indicators above are consistent with those in Figs. 2 and 3. For the signal component with complex IF, the appropriate order of IF estimation can improve the TF accuracy and TF concentration. Increasing the number of the SST for the SST-based methods can significantly improve the TF concentration of the TFR. Therefore, the proposed HMST has advantages over other methods in processing the signal with complex IF. 

## V. EXPERIMENTAL VALIDATION

In this section, we established the validity of the proposed method for bearing fault diagnosis under time-varying rotational speed conditions. The dataset is from the experimental rig of the University of Ottawa [23]. From this rig in Fig. $5 ,$ we can find that two ball bearings (ER16K) support the ends of the rotating shaft, one for failure and one for health, and motor is controlled by ac drive to drive rotating shaft. Rotational speed is measured by the encoder (EPC model 775) and vibration data are collected by accelerometer (Model 623C01). 

The process of the bearing fault diagnosis under timevarying rotational speed conditions is shown in Fig. 6. The size parameters of the bearing (ER16K) are given in Table III, and we can calculate the fault characteristic frequency (FCF) coefficient. The ball-pass frequency of the inner race (BPFI) can be calculated by 


(b)



(a)


![](images/0ad7ea892fb08a96140883d16908ddf80da3ba64709bce8a0c30b76569281326.jpg)



Fig. 5. Test rig.


![](images/3f23bd6772a985ea46be537b767033cc85a528bca43c5e218cfadfd148ecf667.jpg)



Fig. 6. Flowchart for bearing fault diagnosis under time-varying rotational speed conditions.


$$
f _ {i} = \frac {1}{2} n f _ {r} \left[ 1 + \frac {d}{D} \cos \beta \right]\tag{26}
$$

where $f _ { i }$ is the BPFI and $f _ { r }$ is the rotational frequency. The ball-pass frequency of the outer race (BPFO) can be obtained by 

$$
f _ {o} = \frac {1}{2} n f _ {r} \left[ 1 - \frac {d}{D} \cos \beta \right]\tag{27}
$$

where $f _ { o }$ is the BPFO. According to the bearing parameters in Table III, the BFFI and BPFO of the bearing can be calculated as $f _ { i } = 5 . 4 3 f _ { r }$ and $f _ { o } = 3 . 5 7 f _ { r }$ , respectively. Actually, the FCF equals the product of shaft rotational speed and FCF coefficient. Therefore, we can determine the bearing fault type based on the FCF coefficient obtained from the TFA of the bearing fault signal. 

First, the TFA methods are used to analyze the bearing inner-race fault signal. The “I-C-3.mat” is chosen for the analyzed inner-race-fault data and its time interval is [3], [6] s. Fig. 7(a) presents the time waveform of analyzed signal and the signal oscillation caused by the fault shock can be seen, but the type of fault cannot be determined. The rotational frequency is shown in Fig. 7(b), and it first increases and then decreases. The upper envelope of the analyzed signal can be obtained through Hilbert transform, as shown in Fig. 8(a). The STFT TFR is shown in Fig. 8(b), and the component energy in the red solid box area is the highest, which is the principal component of the signal envelope. Fig. 9 gives the enlarge TFRs of this principal component obtained by various TFA methods. It can be found that the TFR resolution of STFT is so low that accurate fault characteristic frequencies cannot be obtained. Compared to the STFT, the TF postprocessing methods have improved the TF performance. Especially, the TFR generated by HMST provides the beat TF readability. Table IV shows the RE values of the TFRs obtained by different TFA methods, and it can be found that the proposed HMST still has the best TF performance. Since the proposed HMST can also be applied to TF filtering by combining with reconstruction algorithms, we use the proposed HMST to obtain the filtered TFR of the principal component and its reconstructed signal, as shown in Fig. 10. By combining the rotational speed and calculating the order spectrum of the reconstructed signal, the fault order of the principal component can be obtained, as shown in Fig. 11. The fault order is 5.46, which is consistent with the theoretical FCF coefficient of the bearing inner-race fault, indicating that the fault type is the bearing inner-race fault. 

![](images/7e528ecc5dca953d7bfd64a4035d88436714c1a8fe324ee201aa5088bd0f9a90.jpg)


![](images/2a7d86d4f50e6fb158c3a25a47e2269ba5086ec5f8fabf23dfbc2985cfc81cfc.jpg)



Fig. 7. (a) Time waveform and (b) rotational frequency.


![](images/bedeecfc34cb1582717dc5ffca23ff5365d6f01cdde1a5ecb2333ec7bf9fe157.jpg)


![](images/3ddb7199304e08d4786b2a51b6f7cb41a09a4232ebcbcf41233e75a6aa0cc05e.jpg)



Fig. 8. (a) Upper envelope and (b) its STFT TFR.


![](images/a671951f1adcfe56c5614eec4bfe0220493c651204bcfca94db91d7009e670af.jpg)


![](images/2379eecc390d59496f9f446dffb027bc3f86c52df09bb23338552e4cc3e3d742.jpg)


![](images/6dca5ced697fac523efe7901232cf057cd71742d36d6c3af2b67cb59be3d2de6.jpg)


![](images/2f326c5741f08bfce413e2808a90fdbfb15377720c88ac531e1921fdce70badc.jpg)


![](images/9f6bdcc0edde573dd1d38a751f46556bf2458551f0f10f8c9ae261752e3bc8a3.jpg)


![](images/84cfee209e072c463cf4d16011aa73403154641a5975fbc0952bb259177a2b27.jpg)



Fig. 9. Enlarge TFR result of the principal component obtained by (a) STFT, (b) SST, (c) SET, (d) MSST, (e) HSST, and (f) HMST.



Time(s)



TABLE IV



RE VALUES FOR VARIOUS TFA METHODS


<table><tr><td>Method</td><td>STFT</td><td>SST</td><td>SET</td><td>MSST</td><td>HSST</td><td>HMST</td></tr><tr><td>RE</td><td>9.247</td><td>6.051</td><td>5.515</td><td>4.367</td><td>5.931</td><td>4.219</td></tr></table>

![](images/fa24c7b846944cd9db80850fe0d4dae45504cf16e6a9c2af144cdc20db4fde6a.jpg)


![](images/480be4351f0141a5c9ab5c9e955ea53822c5b23fd724e3ea4e2121c750cab63e.jpg)



(a)



(b)



Time(s)



Fig. 10. (a) Filtered TFR of the principal component and (b) its reconstructed signal.


![](images/8111a788e54eb8e465d5019cea229285c2955c8190231fe95d33d053f7c5410e.jpg)



Fig. 11. Order spectrum of the reconstructed principal component.


Second, the TFA methods are used to analyze the bearing outer-race fault signal. The “O-C-3.mat” is chosen for the analyzed outer-race-fault data and its time interval is [4.5, 7.5] s. Similar to the bearing inner-race-fault signal, the oscillation caused by the fault shock can also be seen in Fig. 12(a), but the type of bearing fault cannot be determined, and the corresponding rotational frequency is shown in Fig. 12(b). We also use the Hilbert transform to obtain the upper envelope of the analyzed signal in Fig. 13(a), and its STFT TFR is shown in Fig. 13(b). Fig. 14 shows the enlarge TFRs of this principal component obtained by various TFA methods. It can be found that the proposed HMST still has the best TF performance than other TFA methods. We also use the proposed HMST to obtain the filtered TFR of the principal component and its reconstructed signal, as shown in Fig. 15. Then, the fault order of the principal component can be obtained in Fig. 16. The fault order is 3.44, which is consistent with the theoretical FCF coefficient of the bearing outer-race fault, indicating that the fault type is the bearing outer-race fault. 

Finally, the TFA methods are used to analyze the healthy bearing signal as the ablation experiment for the bearing fault diagnosis. The “H-C-3.mat” is chosen for the analyzed healthy bearing data and its time interval is [4.5, 7.5] s. Fig. 17(a) presents the time waveform of the analyzed signal and its rotational frequency is shown in Fig. 17(b). The rotational frequency of the healthy bearing signal is also increases first and then decreases. The upper envelope of the analyzed signal can be obtained through Hilbert transform, as shown in Fig. 18(a). Its STFT TFR is shown in Fig. 18(b), and the component energy in the red solid box area is the highest. 

![](images/b573647cbc5c28d0993a68d9613cdd274363047595f0f822e5549815f812e3bd.jpg)


![](images/846d6a57a70b7eec9e107ffc1fa741d52eb565990c2fafe7eb9354640224c6bd.jpg)



Fig. 12. (a) Time waveform and (b) rotational frequency.


![](images/6519af18e92a49441152a09ba0583b1f426815c1384afb51b2aa2e344c6cc1fc.jpg)


![](images/7b1d4033fc75f5e21576c7dd762351ec70143f2cc33a7fc16754b2fbaa83ecda.jpg)



Fig. 13. (a) Upper envelope and (b) its STFT TFR.


![](images/6d33de76f4b315099c3bbe967662cd1aec5b15444233462e8fbdd04bb92217af.jpg)



(a)



Time(s)


![](images/a4d0ef03788cba941b4c2ecbddab10d26002bbb4c8943eb27af2cb98fc76c309.jpg)



(b)


![](images/f4e1f11fe5ac3510f68d0bfbed1aee25fb09bd95939fd440e55737a27538273d.jpg)



Time(s)



(c)



Time(s)


![](images/936424a1f643382a034acc0f5902a457a8cabfd38a1ab3c35e3f05392f06db56.jpg)



(d)



Time(s)


![](images/5721975c5233f82d4aae60c08a604d35c7d5b71215ada5d632fc8de0a45bff41.jpg)


![](images/7e488d2b60b5cf2a43be65c28d2847ba24f79bea3ce48f5144a6639fb0d3721f.jpg)



(e)



Time(s)



(f)



Time(s)



Fig. 14. Enlarge TFR result of the principal component obtained by (a) STFT, (b) SST, (c) SET, (d) MSST, (e) HSST, and (f) HMST.


![](images/a835245d78481db279b8bc5a141fb9455b401ece565149c1f2b687124ec1fce4.jpg)



(a)


![](images/c4d1395706ab6794b3a6075278ee6b84a7f8aa4f374c1f90b6b047759d4c3cb0.jpg)



Fig. 15. (a) Filtered TFR of the principal component and (b) its reconstructed signal.


![](images/52737bbc08f38ae538eeb2fd5f419b5ae2cfc8c3aab825cddff038e97e5abc60.jpg)



Fig. 16. Order spectrum of the reconstructed principal component.


![](images/f81a0ea7fd9b7113a9c95a3db4a8c5225224603f92ae04744c3aa9d62a0f1e72.jpg)


![](images/79b7dcb9a94b38eb165cc70dd09834d16df4f9b6b65f633b63490dfcde6e42a8.jpg)



Fig. 17. (a) Time waveform and (b) rotational frequency.


![](images/59433f340c46b7e487e457d0049c4175fe2850a46c5b931966436acb714adce3.jpg)


![](images/910c65a5b29b2c441c0320947c4f6850d48c0c98d8e089714a35fad89795e33c.jpg)



(b)



Fig. 18. (a) Upper envelope and (b) its STFT TFR.


Actually, the principal component of the signal envelope is the rotational frequency component. Fig. 19 gives the enlarge TFRs of this principal component obtained by various TFA methods. It can be found that the proposed HMST still has the best TF performance. We also use the proposed HMST to obtain the filtered TFR of the principal component and its reconstructed signal, as shown in Fig. 20(a) and (b), respectively. By combining the rotational speed, the order spectrum of the reconstructed principal component is obtained in Fig. 21. This order is 1, which is consistent with the theoretical rotational frequency coefficient of the bearing, indicating that the analyzed signal has no fault component. We can diagnose this analysis data as health data to verify that the analysis of the two types of faults mentioned above is correct. 

![](images/6778a37ee49ee0e5f17b0015c2ff5999137ce5940b8bf5bb102cf82e7fbebdfd.jpg)



(a)



Time(s)


![](images/29189a3a6b1b6edb4b940e5be67d7a480b87bbb20030e582210f872d114a9d8e.jpg)



(b)



Time(s)


![](images/2fafb6ba0df6fe199c44b7f78737300f9a71c76f19f41953da07ee6fcbc72c35.jpg)


![](images/c3e0170d12ea01bce243889781615399089ca4affb180c6032d20087d813713b.jpg)



(c)



Time(s)



Time(s)



(d)


![](images/a6df799ad59ce25015e659cdb652ffb2118479e1f96277cea5c56f54ca27ee95.jpg)



(e)



Time(s)


![](images/68a366d984ff6fb9e19ef0c5b991acc08fcb5ae4ea14b8a2c8b629632e1ec44f.jpg)



(f)



Time(s)



Fig. 19. Enlarge TFR result of the principal component obtained by (a) STFT, (b) SST, (c) SET, (d) MSST, (e) HSST, and (f) HMST.


![](images/76d8d29943c9d025a57db053eba948b87081fb0cc8ff77f4333522dfd9c5e132.jpg)



(a)


![](images/cf6e84394691f75ad2261ecf920ccc475ebf73bf44b176ab944404aad5ceeaa4.jpg)



(b)



Fig. 20. (a) Filtered TFR of the principal component and (b) its reconstructed signal.


![](images/150799172c9216295b0836c45f488935a6421fa225d414a400742a1e84d32e02.jpg)



Fig. 21. Order spectrum of the reconstructed principal component.


In summary, this research has developed a universal TFA method flow for bearing fault diagnosis, as shown in Fig. 6. Experimental analysis results have shown that the HMST-based TFA method can be applied to bearing fault diagnosis under time-vary speed conditions. 

## VI. CONCLUSION

This article introduced a higher order SST-based TFA method to enhance the TF energy concentration and accuracy, which is termed as HMST. This method can effectively enhance the readability of the TFR through the high-order IF estimation and iterative operations. Numerical and experimental validations illustrated the advantages of the HMST compared with some advanced TFA methods. The comparisons demonstrated that the proposed method is more suitable for processing nonstationary signals with complex time-varying features than other methods. Meanwhile, there are still some limitations that need to be addressed in the proposed method. 

1) How to implement an adaptive strategy for determining the optimal parameters of the HMST. 

2) How to propose a universal TFA method flow for other fault diagnosis by combining the advantages of HMST. 

## APPENDIX

The mathematical induction is used to prove (20). 

1) Show it is true for $M = 2$ 

$$
\begin{array}{l} \mathrm{HMST} ^ {[ 2, N ]} (t, \omega) \\ = \int_ {- \infty} ^ {\infty} \mathrm{HSST} (t, \zeta) \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta \\ = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta (\zeta - \hat {\omega} _ {[ N ]} (t, \eta)) d \eta \\ \quad \times \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \int_ {- \infty} ^ {\infty} \delta \big (\zeta - \hat {\omega} _ {[ N ]} (t, \eta) \big) \\ \quad \times \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta d \eta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \hat {\omega} _ {[ N ]} (t, \eta)) \big) d \zeta d \eta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\omega - \hat {\omega} _ {[ N ]} ^ {[ 2 ]} (t, \eta) \Big) d \eta \quad \text {is True.} \end{array}
$$

2) Assume it is true for $M = k$ 

$$
\begin{array}{l} \text { HMST } ^ {[ k, N ]} (t, \omega) \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\omega - \hat {\omega} _ {[ N ]} ^ {[ k ]} (t, \eta) \Big) d \eta \quad \text { is   True }. \end{array}
$$

Now, prove it is true for $\mathbf { \cdots } M = k + 1 \mathbf { \cdots }$ 

$$
\begin{array}{l} \mathrm{HMST} ^ {[ k + 1, N ]} (t, \omega) \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta (\omega - \hat {\omega} _ {[ N ]} (t, \eta)) d \eta . \end{array}
$$

We know that 

$$
\mathrm{HMST} ^ {[ k, N ]} (t, \xi) = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \left(\xi - \hat {\omega} _ {[ N ]} ^ {[ k ]} (t, \eta)\right) d \eta .
$$

So 

$$
\begin{array}{l} \mathrm{HMST} ^ {[ k + 1 ]} (t, \omega) \\ = \int_ {- \infty} ^ {\infty} \mathrm{HMST} ^ {[ k, N ]} (t, \zeta) \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta \\ = \int_ {- \infty} ^ {\infty} \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\zeta - \hat {\omega} _ {[ N ]} ^ {[ k ]} (t, \eta) \Big) d \eta \\ \quad \times \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \int_ {- \infty} ^ {\infty} \delta \Big (\zeta - \hat {\omega} _ {[ N ]} ^ {[ k ]} (t, \eta) \Big) \\ \quad \times \delta \big (\omega - \hat {\omega} _ {[ N ]} (t, \zeta) \big) d \zeta d \eta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \int_ {- \infty} ^ {\infty} \delta \Big (\omega - \hat {\omega} _ {[ N ]} \Big (t, \hat {\omega} _ {[ N ]} ^ {[ k ]} (t, \eta) \Big) \Big) d \zeta d \eta \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\omega - \hat {\omega} _ {[ N ]} ^ {[ k + 1 ]} (t, \eta) \Big) d \eta \quad \text {is True.} \end{array}
$$

They are the same! So, it is true. 

So 

$$
\begin{array}{l} \text { HMST } ^ {[ M, N ]} (t, \omega) \\ = \int_ {- \infty} ^ {\infty} V _ {x} ^ {w} (t, \eta) \delta \Big (\omega - \hat {\omega} _ {[ N ]} ^ {[ M ]} (t, \eta) \Big) d \eta \quad \text { is   True }. \end{array}
$$

DONE! 

## REFERENCES



[1] G. Yu, “A concentrated time–frequency analysis tool for bearing fault diagnosis,” IEEE Trans. Instrum. Meas., vol. 69, no. 2, pp. 371–381, Feb. 2020. 





[2] W. Bao, F. Li, X. Tu, Y. Hu, and Z. He, “Second-order synchroextracting transform with application to fault diagnosis,” IEEE Trans. Instrum. Meas., vol. 70, pp. 1–9, 2021. 





[3] C. Wang, J. Wang, and X. Zhang, “Automatic radar waveform recognition based on time-frequency analysis and convolutional neural network,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., Mar. 2017, pp. 2437–2441. 





[4] W. Bao, X. Tu, Y. Hu, and F. Li, “Envelope spectrum L-Kurtosis and its application for fault detection of rolling element bearings,” IEEE Trans Instrum. Meas., vol. 69, no. 5, pp. 1993–2002, May 2020. 





[5] S. Wang, I. Selesnick, G. Cai, Y. Feng, X. Sui, and X. Chen, “Nonconvex sparse regularization and convex optimization for bearing fault diagnosis,” IEEE Trans. Ind. Electron., vol. 65, no. 9, pp. 7332–7342, Sep. 2018. 





[6] W. Huang, S. Li, X. Fu, C. Zhang, J. Shi, and Z. Zhu, “Transient extraction based on minimax concave regularized sparse representation for gear fault diagnosis,” Measurement, vol. 151, Feb. 2020, Art. no. 107273. 





[7] W. Huang, G. Gao, N. Li, X. Jiang, and Z. Zhu, “Time-frequency squeezing and generalized demodulation combined for variable speed bearing fault diagnosis,” IEEE Trans. Instrum. Meas., vol. 68, no. 8, pp. 2819–2829, Aug. 2019. 





[8] G. Yu, M. Yu, and C. Xu, “Synchroextracting transform,” IEEE Trans. Ind. Electron., vol. 64, no. 10, pp. 8042–8054, Oct. 2017. 





[9] L.-H. Wang, X.-P. Zhao, J.-X. Wu, Y.-Y. Xie, and Y.-H. Zhang, “Motor fault diagnosis based on short-time Fourier transform and convolutional neural network,” Chin. J. Mech. Eng., vol. 30, no. 6, pp. 1357–1368, Nov. 2017. 





[10] S. Wang et al., “Single and simultaneous fault diagnosis of gearbox via wavelet transform and improved deep residual network under imbalanced data,” Eng. Appl. Artif. Intell., vol. 133, Jul. 2024, Art. no. 108146. 





[11] B. Tang, W. Liu, and T. Song, “Wind turbine fault diagnosis based on Morlet wavelet transformation and wigner-ville distribution,” Renew. Energy, vol. 35, no. 12, pp. 2862–2866, Dec. 2010. 





[12] F. Auger et al., “Time-frequency reassignment and synchrosqueezing: An overview,” IEEE Signal Process. Mag., vol. 30, no. 6, pp. 32–41, Nov. 2013. 





[13] C. Michel and P. Gueguen, “Time-frequency analysis of small frequency variations in civil engineering structures under weak and strong motions using a reassignment method,” Struct. Health Monitor., vol. 9, no. 2, pp. 159–171, Mar. 2010. 





[14] F. Auger and P. Flandrin, “Improving the readability of time-frequency and time-scale representations by the reassignment method,” IEEE Trans. Signal Process., vol. 43, no. 5, pp. 1068–1089, May 1995. 





[15] X. Tu, Q. Zhang, Z. He, Y. Hu, S. Abbas, and F. Li, “Generalized horizontal synchrosqueezing transform: Algorithm and applications,” IEEE Trans. Ind. Electron., vol. 68, no. 6, pp. 5293–5302, Jun. 2021. 





[16] S. Wang, X. Chen, I. W. Selesnick, Y. Guo, C. Tong, and X. Zhang, “Matching synchrosqueezing transform: A useful tool for characterizing signals with fast varying instantaneous frequency and application to machine fault diagnosis,” Mech. Syst. Signal Process., vol. 100, pp. 242–288, Feb. 2018. 





[17] C. Li and M. Liang, “Time–frequency signal analysis for gearbox fault diagnosis using a generalized synchrosqueezing transform,” Mech. Syst. Signal Process., vol. 26, pp. 205–217, Jan. 2012. 





[18] C. Yi, Z. Yu, Y. Lv, and H. Xiao, “Reassigned second-order synchrosqueezing transform and its application to wind turbine fault diagnosis,” Renew. Energy, vol. 161, pp. 736–749, Dec. 2020. 





[19] Y. Hu, X. Tu, and F. Li, “High-order synchrosqueezing wavelet transform and application to planetary gearbox fault diagnosis,” Mech. Syst. Signal Process., vol. 131, pp. 126–151, Sep. 2019. 





[20] G. Yu, Z. Wang, and P. Zhao, “Multisynchrosqueezing transform,” IEEE Trans. Ind. Electron., vol. 66, no. 7, pp. 5441–5455, Jul. 2019. 





[21] I. Daubechies, Y. Wang, and H.-T. Wu, “ConceFT: Concentration of frequency and time via a multitapered synchrosqueezed transform,” Phil. Trans. Roy. Soc. A, Math., Phys. Eng. Sci., vol. 374, no. 2065, Apr. 2016, Art. no. 20150193. 





[22] T. Oberlin, S. Meignen, and V. Perrier, “Second-order synchrosqueezing transform or invertible reassignment? Towards ideal time-frequency representations,” IEEE Trans. Signal Process., vol. 63, no. 5, pp. 1335–1344, Mar. 2015. 





[23] H. Huang and N. Baddour, “Bearing vibration data collected under timevarying rotational speed conditions,” Data Brief, vol. 21, pp. 1745–1749, Dec. 2018. 



![](images/aade8881b22c5f8f8b5972ef809df183d465eb21724c6e79cfed3122d6068243.jpg)


![](images/5b5bfa7273d68edae7f0d400b5b0d15287311863d067bcf26ac97adf399107de.jpg)


Zhen Liu received the B.Eng. degree in mechanical engineering from Hefei University of Technology, Hefei, Anhui, China, in 2015. He is currently pursuing the Ph.D. degree in mechanical engineering with the State Key Laboratory of Mechanical System and Vibration, Shanghai Jiao Tong University, Shanghai, China. 

His research interests include gear dynamics, rotor dynamics, and rotating machinery fault diagnosis. 

Songyong Liu received the Ph.D. degree in mechatronic engineering from China University of Mining and Technology, Xuzhou, China, in 2009. 

He is currently a Full Professor with the School of Mechatronic Engineering, China University of Mining and Technology. His research interests include machine learning, special robots, and automatic systems. 

![](images/b57d37a5eb2b204e7a5894c8666802f9795136110b8e844f24f2e9e53ea62446.jpg)



Wenjie Bao received the Ph.D. degree in mechanical engineering from Shanghai Jiao Tong University, Shanghai, China, in 2023.



He is currently a Lecturer with the School of Mechatronic Engineering, China University of Mining and Technology, Xuzhou, China. His current research interests include signal processing and machine health diagnosis.


![](images/ba54c5f049ad1e9633dd6738526cf97eb2df55b99f90798ea33861f323396b7f.jpg)


Fucai Li received the B.Eng. and Ph.D. degrees in mechanical engineering from Xi’an Jiaotong University, Xi’an, China, in 1998 and 2003, respectively. 

He is currently a Professor with Shanghai Jiao Tong University, Shanghai, China. His current research interests include structural health monitoring, fault diagnosis for mechanical systems, sensing technology, and digital signal processing. 