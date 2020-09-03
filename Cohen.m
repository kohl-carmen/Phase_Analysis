%% Mike X Cohen wrote all this stuff
%% mikexcohen@gmail.com
% This is just a collection of bits I took from his youtube videos about
% inter-phase clustering
% (I probably commented/edited quite a lot but didn't track, not sure)



%% time frequency analysis using morlet wavelet
%https://www.youtube.com/watch?v=wgRgodvU_Ms&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=2
%btw, i=sqrt(-1) (doens't really exist as a number), i always leads to complex numbers

%morlet params
srate=600;%sampling rate in Hz
time=-2:1/srate:2; %wavelet time, not data time  - have time of 0 at centre of wavelet, so have an odd number of time points
%wavelet sampling rate MUST be the same as data sampling rate
frex=6.5; % frequency of this wavelet

%create wavelet
sine_wave=exp(1i*2*pi*frex.*time); %this is just a sine wave at my given frequency - it's actally a complex sine wave because that's what allows us to extract power and phase
s=7/(2*pi*frex);% this is the standard deviation of the gaussian -> 7 is the nnumber of cycles!!
%number of cycles: the larger the nr cycles, the worse the time precision and the better the freq precision - stay within 3 to 10 https://www.youtube.com/watch?v=fkQGnYUv-FI&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=6
gaus_win=exp((-time.^2)./(2*s^2)); %this is the gaussian
cmw=sine_wave .* gaus_win; %multiply sine and gauss to get wavelet

%plot wavelet (it has a real part and an imaginary part (imaginary is sine
%and real is cosine part so they're shiften by 90 degrees)
figure 
plot(time,real(cmw))
plot(time,imag(cmw))

clf
plot3(time,real(cmw),imag(cmw),'linew',2)
axis image
xlabel('Time'), ylabel('Real part'), zlabel('Imaginary part')
rotate3d

% that's morelt wavelets in the time domain. in the frequency domain,
% they're gaussian shaped
% so to look at it in the freq domain,  we're taking a fast fourier transform of the wavelet
cmwX=fft(cmw)/length(cmw); % this is the morlet wavelet in the freq domain (the more cycles, the wider the gaussian here)
hz=linspace(0,srate/2,floor(length(cmw)/2)+1);
clf
plot(hz,2*abs(cmwX(1:length(hz))))

%https://www.youtube.com/watch?v=4TTpwIZrUAo&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=3
% convolution in the time domain is equivalent to multiplication in the
% freq domain -> so in practice, we'll only ever do freq domain stuff cause
% faster
% wavelet convolution via frequency domain mutliplication:
% 1) Take fft of signal
% 2) mutliply fourier spectra point by point
% 3) take inverse fft

% so lets take some data and apply this
data=X{1}(:,1)'; % this is based on data processed by the spectral events toolbox because that's what I happened to be using at the time. Use any EEG data here.
%create wavelet like above like above
srate=600;
time=-2:1/srate:2;
frex=6.5; 
sine_wave=exp(1i*2*pi*frex.*time);
s=7/(2*pi*frex);
gaus_win=exp((-time.^2)./(2*s^2)); 
cmw=sine_wave .* gaus_win; 

%now define convolution parameters
nData=length(data);%length of data
nKern=length(cmw);%length of wavelet
nConv=nData+nKern-1;%length of conolution outcome is always  data+wave-1

%now take fft of morlet wavelet
cmwX=fft(cmw,nConv);%by gigving it the n of the outcome, it does the zero padding itself
%and amplitude-normalise in the frequency domain (amplitude scale morlet wvelet by its peak)
cmwX=cmwX./max(cmwX);%not completely necessary but makes sure that the outcome is gonnabe in the original units of the signal

%now take fft of data
dataX=fft(data,nConv);

%now do convolution
conv_res=cmwX.*dataX;  %because conv is mutliplication in freq fomain and we're now in freq domain(becuase we took fft of everything)
%also, usually we want to put this back into time domain after, so we can
%do this in one step conv_res=ifft(cmwX.*dataX), but here we're doing it
%step by step

%compute hz for plotting
hz=linspace(0,srate/2,floor(length(cmw)/2)+1);
figure
hold on
plot(hz,2*abs(dataX(1:length(hz))/length(data)))%plots freq of data
plot(hz,abs(cmwX(1:length(hz))))%plots wavelet in freq
plot(hz,2*abs(conv_res(1:length(hz))/length(data))); %plots producut of multiplying those two

%now lets get it back into the time domain
%keep in mind, the length of the result of the convolution is longer than
%the signal
%so cut 1/2 of the length of the wavelet from beginning and end
half_wav=floor(length(cmw)/2)+1;


%take inverse fourier
conv_res_timedomain=ifft(conv_res);
conv_res_timedomain=conv_res_timedomain(half_wav-1:end-half_wav); %cut sides off (not data)

figure
plot(tVec,data,'k')
hold on
plot(tVec,real(conv_res_timedomain),'r')




%% get power and phase
%https://www.youtube.com/watch?v=A4M0cZSrHzY&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=4
% we're using a complex morelt wvelet, so it has a real and and an
% imaginary part (so thats just sine ans cosine, so a bit shifted). so when
% we convolute that, for each timepoint, we get a single complex number.
% complex numbers can be represented in a coord system, where the y
% axis is the imaginary axis and the x axis i the real axis. then we
% can plot that complex number as a point in that space. Now if i plot the
% vector from the origin to that point, the length of the vector is
% amplitude and the angle of that vector is phase angle.
% so at the end of an analysis, we get a single complex number for each
% time point and each frequency (obv only if we have a wavelet for each
% frequency).the filtered EEG signal is the real part of those complex 
% numbers

% so let's use data and wavelet from above

as=conv_res_timedomain;

% lets plot the result from earlier in some new ways
figure(1), clf %this is the whole result of the convolution with imaginary and real parts over time
plot3(tVec,real(as),imag(as),'k')
xlabel('Time (ms)'), ylabel('real part'), zlabel('imaginary part')
rotate3d

figure(2), clf %this is the phase angle over time (and also ampliutde, both axis)
plot3(tVec,abs(as),angle(as),'k')
%2D: plot(tVec,angle(as),'k') % just phase)
xlabel('Time (ms)'), ylabel('Amplitude'), zlabel('Phase')
rotate3d

figure(3), clf
% plot the filtered signal (projection onto real axis)
subplot(311)
plot(tVec,real(as))
xlabel('Time (ms)'), ylabel('Amplitude (\muV)')

% angle(as) gives you phase per timepoints (for this particular freq)
% as is the timedomain result of the convultion

% Now if you have many trials:
% could do
% for freq=1:Freq
%   for trial=1:Trial
%       convolution
%   end
% end

% but we can actually get rid ot the trial loop!
% we can jsut concatenate all the trials (string them onto each other into
% one big trial), do our thing, and then split it up again, and then avg
% them. same result as doing each trial separately
%https://www.youtube.com/watch?v=wdrXzqgcYLM&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=5

%% this is the slow version with the trial loop
data=X{1};
% frequency parameters
min_freq =  2;
max_freq = 30;
num_frex = 40;
frex = linspace(min_freq,max_freq,num_frex);

% other wavelet parameters
range_cycles = [ 4 10 ];

s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);
wavtime = -2:1/srate:2;
half_wave = (length(wavtime)-1)/2;

tic; % start matlab timer

% FFT parameters
nWave = length(wavtime);
nData = size(data,1);
nConv = nWave + nData - 1;

% initialize output time-frequency data
tf = zeros(length(frex),size(data,1),size(data,2));

% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    % the wavelet doesn't change on each trial...
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    % now loop over trials...
    for triali=1:size(data,2)
        
        dataX = fft(data(:,triali)', nConv);
        
        % run convolution
        as = ifft(waveletX .* dataX);
        as = as(half_wave+1:end-half_wave);
    
        % put power data into big matrix
        tf(fi,:,triali) = abs(as).^2;
    end
end

tfTrialAve = mean(tf,3);

computationTime = toc; 

% plot results
figure(1), clf
contourf(tVec,frex,tfTrialAve,40,'linecolor','none')

%% this is the same but more efficient (no trial loop)
% we make one giant trial
data=X{1};
tic; % restart matlab timer

% FFT parameters
nWave = length(wavtime);
nData = size(data,1) * size(data,2); % This line is different from above!!
nConv = nWave + nData - 1;

% initialize output time-frequency data
tf = zeros(length(frex),size(data,1));

% now compute the FFT of all trials concatenated
alldata = reshape( data ,1,[]);
dataX   = fft( alldata ,nConv );


% loop over frequencies
for fi=1:length(frex)
    
    % create wavelet and get its FFT
    % the wavelet doesn't change on each trial...
    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
    waveletX = fft(wavelet,nConv);
    waveletX = waveletX ./ max(waveletX);
    
    % now run convolution in one step
    as = ifft(waveletX .* dataX);
    as = as(half_wave+1:end-half_wave);
    
    % and reshape back to time X trials
    as = reshape( as, size(data,1), size(data,2) );
    
    % compute power and average over trials
    tf(fi,:) = mean( abs(as).^2 ,2);
end

computationTime(2) = toc; % end matlab timer

% plot results
figure(2), clf
contourf(tVec,frex,tf,40,'linecolor','none')
 
%the only diffference between those two comes from edge artifacts




%% nr of wavelet cycles
% ok now we'll use wavelets with different numbers of cycels for
% the same analysis, just to demonstrate that it matters

%number of cycles: the larger the nr cycles, the worse the time precision
%and the better the freq precision 

data=X{1};
% wavelet parameters
num_frex = 40;
min_freq =  2;
max_freq = 30;

% set a few different wavelet widths ("number of cycles" parameter)
num_cycles = [ 2 6 8 15 ]; %ths is a bit extreme. stay between 3 and 10

% other wavelet parameters
frex = linspace(min_freq,max_freq,num_frex);
time = -2:1/srate:2;
half_wave = (length(time)-1)/2;

% FFT parameters
nKern = length(time);
nData = size(data,1)*size(data,2);
nConv = nKern+nData-1;

% initialize output time-frequency data
tf = zeros(length(num_cycles),length(frex),size(data,1));


% FFT of data (doesn't change on frequency iteration)
dataX = fft(reshape(data,1,[]),nConv);

% loop over cycles
for cyclei=1:length(num_cycles) %so now  we're doing all of this for diffferent numbers of cycles
    
    for fi=1:length(frex)
        
        % create wavelet and get its FFT
        s = num_cycles(cyclei)/(2*pi*frex(fi));
        
        cmw  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
        cmwX = fft(cmw,nConv);
        cmwX = cmwX./max(cmwX);
        
        % run convolution, trim edges, and reshape to 2D (time X trials) 
        as = ifft(cmwX.*dataX,nConv);
        as = as(half_wave+1:end-half_wave);
        as = reshape(as,size(data,1),size(data,2));
        
        % put power data into big matrix
        tf(cyclei,fi,:) = mean(abs(as).^2,2);
    end
    
     
end

% plot results
figure(3), clf
for cyclei=1:length(num_cycles)
    subplot(2,2,cyclei)
    
    contourf(tVec,frex,squeeze(tf(cyclei,:,:)),40,'linecolor','none')
%     set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
    title([ 'Wavelet with ' num2str(num_cycles(cyclei)) ' cycles' ])
    xlabel('Time (ms)'), ylabel('Frequency (Hz)')
end


%% use variable number of wavelet cycles

% set a few different wavelet widths (number of wavelet cycles)
range_cycles = [ 4 10 ];%min and max cycls

% other wavelet parameters
nCycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);%get 40  diff nr of cycles, logarithymically scaled
% so now, we have a different nr of cycles for each freq we defined
% earlier. so in the freq loop, it makes a new wavelet and grabs this nr of
% cycles out of here

% initialize output time-frequency data
tf = zeros(length(frex),size(data,1));

for fi=1:length(frex)
    
    % create wavelet and get its FFT
    s = nCycles(fi)/(2*pi*frex(fi));
    cmw = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
    
    cmwX = fft(cmw,nConv);
    
    % run convolution
    as = ifft(cmwX.*dataX,nConv);
    as = as(half_wave+1:end-half_wave);
    as = reshape(as,size(data,1),size(data,2));
    
    % put power data into big matrix
    tf(fi,:) = mean(abs(as).^2,2);
end


% plot results
figure(4), clf
subplot(2,2,1)

contourf(tVec,frex,tf,40,'linecolor','none')
% set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
title('Convolution with a range of cycles')
xlabel('Time (ms)'), ylabel('Frequency (Hz)')




%% inter-trial phase coherence

data=trough_lock';%whatever data

% data=rnd_50';
% wavelet parameters
num_frex = 40;
min_freq =  2;
max_freq = 30;

% set range for variable number of wavelet cycles
range_cycles = [ 4 10 ];

% other wavelet parameters
frex = logspace(log10(min_freq),log10(max_freq),num_frex);
wavecycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
time = -2:1/srate:2;
half_wave_size = (length(time)-1)/2;

% FFT parameters
nWave = length(time);
nData = size(data,1)*size(data,2);
nConv = nWave+nData-1;

% FFT of data (doesn't change on frequency iteration)
dataX = fft( reshape(data,1,nData) ,nConv);

% initialize output time-frequency data
tf = zeros(num_frex,size(data,1));

% loop over frequencies
for fi=1:num_frex
    
    % create wavelet and get its FFT
    s = wavecycles(fi)/(2*pi*frex(fi));
    wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
    waveletX = fft(wavelet,nConv);
    
    % run convolution
    as = ifft(waveletX.*dataX,nConv);
    as = as(half_wave_size+1:end-half_wave_size);
    as = reshape(as,size(data,1),size(data,2));
    
    % compute ITPC
    tf(fi,:) = abs(mean(exp(1i*angle(as)),2)); %this transforms our phase angles into vectors in a polar plane - then takes the avergae of those vectors, and then the length of that average vector
%     angles_all(fi,:,:)=angle(as);
end

% plot results
figure(1), clf
contourf(1:61,frex,tf,40,'linecolor','none')
% set(gca,'clim',[0 .6],'ydir','normal','xlim',[-300 1000])
title('ITPC')

angledata=squeeze(angles_all(1,1,:));

% compute ITPC and preferred phase angle
itpc      = abs(mean(exp(1i*angledata)));
prefAngle = angle(mean(exp(1i*angledata)));


% and plot...
figure(2), clf

% as linear histogram
subplot(3,3,4)
hist(angledata,20)
xlabel('Phase angle'), ylabel('Count')
set(gca,'xlim',[0 2*pi])
title([ 'Observed ITPC: ' num2str(itpc) ])

% and as polar distribution
subplot(1,2,2)
polar([zeros(1,200); angledata'],[zeros(1,200); ones(1,200)],'k')
hold on
h = polar([0 prefAngle],[0 itpc],'m');
set(h,'linew',3)
title([ 'Observed ITPC: ' num2str(itpc) ])



    

%% using hilbert transform (alternative to wavelet)
% hilbert might be useful because it looks at a whole band at a time
% https://www.youtube.com/watch?v=VyLU8hlhI-I
% generate random numbers
n = 21;
randomnumbers = randn(n,1);

% take FFT
f = fft(randomnumbers);
% create a copy of the fourir coefficients that is multiplied by the complex operator
complexf = 1i*f;

% find indices of positive and negative frequencies
posF = 2:floor(n/2)+mod(n,2);
negF = ceil(n/2)+1+~mod(n,2):n;

% rotate Fourier coefficients
% (note 1: this works by computing the iAsin(2pft) component, i.e., the phase quadrature)
% (note 2: positive frequencies are rotated counter-clockwise; negative frequencies are rotated clockwise)
f(posF) = f(posF) + -1i*complexf(posF);
f(negF) = f(negF) +  1i*complexf(negF);
% The next two lines are an alternative and slightly faster method. 
% The book explains why this is equivalent to the previous two lines.
% f(posF) = f(posF)*2;
% f(negF) = f(negF)*0;

% take inverse FFT
hilbertx = ifft(f);

% that so far is the same as using the matlab hilbert function in the
% signal processing toolbox
% compare with Matlab function hilbert 
hilbertm = hilbert(randomnumbers);


% plot results
figure(1), clf
subplot(211)
plot(abs(hilbertm))
hold on
plot(abs(hilbertx),'ro')
set(gca,'xlim',[.5 n+.5])
legend({'Matlab Hilbert function';'"manual" Hilbert'})
title('magnitude of Hilbert transform')

subplot(212)
plot(angle(hilbertm))
hold on
plot(angle(hilbertx),'ro')
set(gca,'xlim',[.5 n+.5])
legend({'Matlab Hilbert function';'"manual" Hilbert'})
title('phase of Hilbert transform')

% try with real data
load sampleEEGdata

% ERP, and its hilbert transform
erp  = squeeze(mean(EEG.data(48,:,:),3));
erpH = hilbert(erp);

figure(2), clf

% plot ERP and real part of Hilbert transformed ERP
subplot(311)
plot(EEG.times,erp), hold on
plot(EEG.times,real(erpH),'r')
legend({'ERP';'real(hilbert(erp))'})

% plot ERP and magnitude
subplot(312)
plot(EEG.times,erp), hold on
plot(EEG.times,abs(erpH),'r')
legend({'ERP';'abs(hilbert(erp))'})

% plot ERP and phase angle time series
subplot(313)
plot(EEG.times,erp), hold on
plot(EEG.times,angle(erpH),'r')
legend({'ERP';'angle(hilbert(erp))'})
xlabel('Time (ms)'), ylabel('Voltage or radians')


% plot as 3d line
figure(3), clf
plot3(EEG.times,real(erpH),imag(erpH))
xlabel('Time (ms)'), ylabel('Real part'), zlabel('Imaginary part')
axis tight
rotate3d

% these phases look a mess because it's broadband signal
% -> bandpass filter FIRST
%https://www.youtube.com/watch?v=ljw3gW-nL0E

% load sampleEEGdata

% specify Nyquist freuqency
nyquist = EEG.srate/2;

% filter frequency band
filtbound = [4 10]; % Hz

% transition width
trans_width = 0.2; % fraction of 1, thus 20%

% filter order
filt_order = round(3*(EEG.srate/filtbound(1)));

% frequency vector (as fraction of Nyquist
ffrequencies  = [ 0 (1-trans_width)*filtbound(1) filtbound (1+trans_width)*filtbound(2) nyquist ]/nyquist;

% shape of filter (must be the same number of elements as frequency vector
idealresponse = [ 0 0 1 1 0 0 ];

% get filter weights
filterweights = firls(filt_order,ffrequencies,idealresponse);

% plot for visual inspection
figure(1), clf
subplot(211)
plot(ffrequencies*nyquist,idealresponse,'k--o','markerface','m')
set(gca,'ylim',[-.1 1.1],'xlim',[-2 nyquist+2])
xlabel('Frequencies (Hz)'), ylabel('Response amplitude')

subplot(212)
plot((0:filt_order)*(1000/EEG.srate),filterweights)
xlabel('Time (ms)'), ylabel('Amplitude')

% apply filter to data
filtered_data = zeros(EEG.nbchan,EEG.pnts);
for chani=1:EEG.nbchan
    filtered_data(chani,:) = filtfilt(filterweights,1,double(EEG.data(chani,:,1)));
end

figure(2), clf
plot(EEG.times,squeeze(EEG.data(47,:,1)))
hold on
plot(EEG.times,squeeze(filtered_data(47,:)),'r','linew',2)
xlabel('Time (ms)'), ylabel('Voltage (\muV)')
legend({'raw data';'filtered'})


% compare three transition widths
nyquist    = EEG.srate/2;
filtbond   = [ 7 12 ];
t_widths   = [.1 .15 .2];
filt_order = round(3*(EEG.srate/filtbond(1)));

idealresponse = [ 0 0 1 1 0 0 ];

ffrequencies  = zeros(3,6);
filterweights = zeros(3,filt_order+1);

% frequency vector (as fraction of Nyquist)
for i=1:3
    ffrequencies(i,:)  = [ 0 (1-t_widths(i))*filtbond(1) filtbond (1+t_widths(i))*filtbond(2) nyquist ]/nyquist;
    filterweights(i,:) = firls(filt_order,ffrequencies(i,:),idealresponse);
end

% plot
figure(4), clf
subplot(211)
plot((1:filt_order+1)*(1000/EEG.srate),filterweights')
xlabel('time (ms)')
title('Time-domain filter kernel')

filterFreqDomain = abs(fft(filterweights,[],2));
frequenciesHz    = linspace(0,nyquist,floor(filt_order/2)+1);
subplot(212)
plot(frequenciesHz,filterFreqDomain(:,1:length(frequenciesHz)))
set(gca,'xlim',[0 60],'ylim',[-.1 1.2])
xlabel('Frequencies (Hz)')
title('Frequency-domain filter kernel')
legend({'filter 10%','filter 15%','filter 20%'})

% compute and plot power
chan4filt = strcmpi('o1',{EEG.chanlocs.labels});
baseidx   = dsearchn(EEG.times',[-400 -100]');

pow = zeros(3,EEG.pnts);

for i=1:3
    filtered_data = reshape(filtfilt(filterweights(i,:),1,double(reshape(EEG.data(chan4filt,:,:),1,[]))),EEG.pnts,EEG.trials);
    
    temppow  = mean(abs(hilbert(filtered_data)).^2,2);
    pow(i,:) = 10*log10( temppow./mean(temppow(baseidx(1):baseidx(2))) );
end

figure(5), clf
plot(EEG.times,pow)
xlabel('Time (ms)'), ylabel('power (dB)')
legend({'filter 10%','filter 15%','filter 20%'})
set(gca,'xlim',[-300 1200])



    
    

