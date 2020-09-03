%% ITPC analysis 
% see Cohen.m for a breakdown of the ITPC part
% Using Shin's MEG detection data
% This script takes beta events closest to tap and finds the phase angle at
% 0 (=tap) to see if there's any phase clustering, comparing detected and
% non-detected conditions

clear
Partic=1:10;
ppt=0;

if ppt
    h = actxserver('PowerPoint.Application');
    Presentation = h.Presentation.Add;
end

KEEP_ITPC=struct();
KEEP_ANGLE=struct();

for partic=1:length(Partic)

    %% load data
    data_path='F:\Brown\Shin Data\HumanDetection\';
    filename=strcat('prestim_humandetection_subject',num2str(Partic(partic)),'.mat');

    load(strcat(data_path,filename))

    % Fs: sampling rate (Hz)
    % prestim__yes_no: 1 second prestimulus trace. 
    % prestim_TFR_yes_no: 1 second prestimulus time-frequency representation (TFR). 1st dimension is frequency, 2nd dimension is time, 3rd dimension is trials
    % fVec: frequeTFRncy vector corresponding to 1st dimension in prestim_TFR_yes_no (Hz)
    % tVec: time vector corresponding to 2nd dimension in prestim_TFR_yes_no (ms)
    % YorN: behavior outcome of each trial corresponding to 3rd dimension in prestim_TFR_yes_no.
    % YorN==1 trials correspond to detected trials and YorN==0 trials correspond to non-detected trials

    data=prestim_raw_yes_no';
    nr_trials=size(data,2);
    beh=YorN;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Define Events
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Fs=600;
    dt=1000/600;
    X{1}=data;
    tVec_assumed=linspace(1/Fs,1,Fs);
    betaband=[15 29];
    
    %% get beta events
    ryan='C:\Users\ckohl\Documents\MATLAB\Ryan';
    addpath(ryan)
    eventBand=betaband;
    fVec=3:3:30;
    findMethod=1;
    vis=0;
    classLabels{1}=1;
    tVec_assumed=linspace(1/Fs,1,Fs);
    [specEv_struct,TFRs,X] = spectralevents(eventBand,fVec,Fs,findMethod,vis,X,classLabels);

    fprintf('\n\nPartic: %d \nTrials: %d \nEvents: %d\n',partic,nr_trials,size(specEv_struct.Events.Events.maximapower,1))
    plott=1;   
    time_interval=100;% how long around maxfre to look for trough
    plot_time=100;
    trough_lock=nan(size(X{1}));
    trough_is=nan(nr_trials,1);
    %last_beta_before_tap
    no_beta_trials=[];
    for trial = 1:nr_trials
        if any(trial==specEv_struct.Events.Events.trialind)
            last_beta_i=max(find(specEv_struct.Events.Events.trialind==trial));
            
            max_t=specEv_struct.Events.Events.maximatiming(last_beta_i);
            max_t_realtime=tVec(find(round(tVec_assumed,3)==round(max_t,3)));
            
            [trough,trough_i]=min(X{1}(max(1,find(tVec==max_t_realtime)-time_interval/2/dt):min(find(tVec==max_t_realtime)+time_interval/2/dt,length(tVec)),trial));  
            temp=1:length(tVec);
            temp=temp(max(1,find(tVec==max_t_realtime)-time_interval/2/dt):min(find(tVec==max_t_realtime)+time_interval/2/dt,length(tVec)));
            trough_i=temp(trough_i);
            
            %time to look prev to trough
            t_pre=50;
            if trough_i>t_pre/dt
                trough_data_i=trough_i-t_pre/dt : length(tVec);
                
                trough_is(trial)=trough_i;
                trough_lock(trial,trough_data_i)=X{1}(trough_data_i,trial)';
                if plott
%                     subplot(10,20,trial)
                    subplot(5,1,[1:4])
                    hold on
%                     title(strcat('Trial',num2str(trial)))
                    title(strcat('Partic',num2str(partic)))
                    plot(tVec, trough_lock(trial,:),'Color',[.5 .5 .5])
                    plot(tVec(trough_i), trough_lock(trial,trough_i),'ro')
                end
            else
                no_beta_trials=[no_beta_trials,trial];
            end
        else
            no_beta_trials=[no_beta_trials,trial];
        end
    end
    if plott
        subplot(5,1,5)
        h=histogram(tVec(trough_is(~isnan(trough_is))));
        h.FaceColor=[.5 .5 .5];
        xlim([tVec(1) tVec(end)])
    end

  
    
    %delete trials with no beta
    trough_lock=trough_lock(1:size(X{1},2),:);
    trough_is(nansum(trough_lock,2)==0)=[];    
    beh(nansum(trough_lock,2)==0)=[];    
    trough_lock(nansum(trough_lock,2)==0,:)=[];   
    
    og_trough_lock=trough_lock;
    og_trough_is=trough_is;
    % chop off ends to get rid of nan (this will cut off betas but who
    % cares
    % let's say I want at least 100ms
    cutoff=100;
    [temp,cutoff_i]=min(abs(tVec-(tVec(end)-cutoff)));
    trough_lock=trough_lock(:,cutoff_i:end);
    %delete though trials where not enough samples are in cutoff
    delete=[];
    for i=1:size(trough_lock,1)
        if sum(trough_lock(i,1:5))==0
            delete=[delete,i];
        end
    end
    trough_lock(delete,:)=[];
    trough_is(delete)=[];
    beh(delete)=[];
    og_trough_lock(delete,:)=[];
    og_trough_is(delete,:)=[];
    tVec_new=tVec(cutoff_i:end);
      
    
    %% only keep if beta within X of tap
    beta_distance=1000;
    keepi=find(tVec(trough_is)>=tVec(end)-beta_distance);
    trough_lock=trough_lock(keepi,:);
    trough_is=trough_is(keepi);
    og_trough_lock=og_trough_lock(keepi,:);
    beh=beh(keepi);
    subplot(5,1,[1:4])
    hold on
    title(strcat('Partic',num2str(partic), '-Remaining'))
    for trial=1:size(trough_lock,1)
        plot(tVec_new, trough_lock(trial,:),'Color','b')
        plot(tVec(trough_is(trial)), og_trough_lock(trial,trough_is(trial)),'bo')
    end
    if ppt
        print('-dpng','-r150',strcat('temp','.png'));
        blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
        Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
        Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
    end
    
        
    %split intp conds
    detect=trough_lock(beh==1,:);
    non=trough_lock(beh==0,:);
    fprintf('\nTrials without events: %d\nTrials too short:  %d\n',length(no_beta_trials),length(delete))
    fprintf('\n\n-----\nRemaining:\nDetected:  %d\nNon-detected:  %d\n-----\n',size(detect,1),size(non,1))
    
    Conds={'detect','non'};
    for conds=1:2
        data=eval(Conds{conds});
        data=data';

    %% get phase
    %TIME x TRIAL
%     data=trough_lock';
% 
%      data=rnd_50';
%     data=trough_lock';
    % wavelet parameters
    num_frex = 15;
    min_freq =  8;
    max_freq = 29;
    srate=Fs;


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
    angles_all= zeros(num_frex,size(data,1),size(data,2));

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
        angles_all(fi,:,:)=angle(as);
    end

    % plot results
    figure(1), 
    subplot(1,2,conds)
    hold on
%     contourf(1:size(tf,2),frex,tf,40,'linecolor','none')
    try
        contourf(tVec_new,frex,tf,40,'linecolor','none')
    catch
        contourf(1:size(tf,2),frex,tf,40,'linecolor','none')
    end
    set(gca,'clim',[0 .5])
%     cbar
    if conds==1
        title(strcat('ITPC Detected  (',num2str(size(detect,1)),')'))
    else
        title(strcat('ITPC Nondetected (',num2str(size(non,1)),')'))
    end
  
    if conds==2 & ppt
        print('-dpng','-r150',strcat('temp','.png'));
        blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
        Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
        Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
    end

    %angles_all= freq x time x trial
    figure('units','normalized','outerposition', [0 0 1 1]);
    time_oi=58;
    for freq_oi=1:length(frex)
        angledata=squeeze(angles_all(freq_oi,time_oi,:));

        % compute ITPC and preferred phase angle
        itpc      = abs(mean(exp(1i*angledata)));
        prefAngle = angle(mean(exp(1i*angledata)));


%         % and plot...
%         figure(2), clf
% 
%         % as linear histogram
%         subplot(3,3,4)
%         hist(angledata,20)
%         xlabel('Phase angle'), ylabel('Count')
%         set(gca,'xlim',[0 2*pi])
%         title([ 'Observed ITPC: ' num2str(itpc) ])

        % and as polar distribution
        subplot(4,4,freq_oi)
        u=polar([zeros(1,size(angles_all,3)); angledata'],[zeros(1,size(angles_all,3)); ones(1,size(angles_all,3))]);;
        set(u,'Color',[.5 .5 .5]);
        hold on
        h = polar([0 prefAngle],[0 itpc],'r');
        set(h,'linew',4)
        if conds==1
            tit=sprintf('Freq: %2.2f\nITPC-Detected: %2.2f',frex(freq_oi), itpc);
        else
            tit=sprintf('Freq: %2.2f\nITPC-NonDetected: %2.2f',frex(freq_oi), itpc);
        end
        title(tit)
    end
    
    % ITPC per timepoint
    %angles_all= freq x time x trial
    figure('units','normalized','outerposition', [0 0 1 1]);
    time_oi=58;
    for time_oi=1:size(angles_all,2)
        for freq_oi=1:length(frex)
            angledata=squeeze(angles_all(freq_oi,time_oi,:));

            % compute ITPC and preferred phase angle
            itpc      = abs(mean(exp(1i*angledata)));
            prefAngle = angle(mean(exp(1i*angledata)));

            KEEP_ITPC.(Conds{conds})(partic,freq_oi,time_oi)=itpc;
            KEEP_ANGLE.(Conds{conds})(partic,freq_oi,time_oi)=prefAngle;
        end
    end
    if ppt
        print('-dpng','-r150',strcat('temp','.png'));
        blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
        Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
        Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
    end
    TF_ALL.(Conds{conds})(:,:,partic)=tf;
    ANGLES_ALL.(Conds{conds}).(strcat('P',num2str(partic)))=angles_all;
    end
    close all
end


%% PLOT ITPC per partic,freq,time
figure
clf
c=parula(length(Partic));
for conds=1:length(Conds)  
    subplot(1,2,conds)
    for freq=1:length(frex)
          for partic=1:length(Partic)
        
                a=scatter3(tVec_new, ones(size(tVec_new)).*frex(freq), squeeze(KEEP_ITPC.(Conds{conds})(partic,freq,:)));
                a.Marker='.';
                a.MarkerEdgeColor=c(partic,:);
                
                hold on
        end
    end
    title(Conds{conds})
    xlabel('Time')
    ylabel('Frequency')
    zlabel('ITPC')
end
if ppt
    print('-dpng','-r150',strcat('temp','.png'));
    blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
    Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
    Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
end
            
        
        

 
%                 plot(tVec_new,squeeze(KEEP_ITPC.(Conds{conds})(partic,freq,:)),'-','Color',c(partic,:))
%             
%             all partics in one, each partic diff colour
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% % %% So this cohen bit now works but not with nans, so let's try something else
% % 
% % 
% % %% if we did just norlet wavelet
% % %https://www.youtube.com/watch?v=wgRgodvU_Ms&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=2
% % %btw, i=sqrt(-1) (doens't really exist as a number), i always leads to
% % %cmplex numbers
% % %morlet params
% % srate=600;%sampling rate in Hz
% % time=-2:1/srate:2; %wavelet time, not data time  - have time of 0 at centre of wavelet, so have an odd number of time points
% % %wavelet smaoling rate MUST be the same as data sampling rate
% % frex=6.5; % frequency of this wavelet
% % 
% % %create wavelet
% % sine_wave=exp(1i*2*pi*frex.*time); %this is just a sine wave at my given frequency - it's actally a complex sine wave because that's what allows us to extract power and phase
% % s=7/(2*pi*frex);% this is the standard deviation of the gaussian -> 7 is the nnumber of cycles!!
% % %number of cycles: the larger the nr cycles, the worse the time precision
% % %and the better the freq precision (so maybe use fewer cycles for time
% % %precision of phase??) stay within 3 to 10 https://www.youtube.com/watch?v=fkQGnYUv-FI&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=6
% % gaus_win=exp((-time.^2)./(2*s^2)); %this is the gaussian
% % cmw=sine_wave .* gaus_win; %multiply sine and gauss to get wavelet
% % 
% % %plot wavelet (it has a real part and an imaginary part (imaginary is sine
% % %and real is cosine part so theyre shiften by 90 degrees)
% % figure 
% % plot(time,real(cmw))
% % plot(time,imag(cmw))
% % 
% % clf
% % plot3(time,real(cmw),imag(cmw),'linew',2)
% % axis image
% % xlabel('Time'), ylabel('Real part'), zlabel('Imaginary part')
% % rotate3d
% % 
% % % that's morelt wavelets in the time domain. in the frequency domain,
% % % they're gaussian shaped
% % 
% % % so we're taking a fast fourier transform of the wavelet
% % cmwX=fft(cmw)/length(cmw); % this is the morlet wavelet in the freq domain (the more cycles, the wider the gaussian here)
% % hz=linspace(0,srate/2,floor(length(cmw)/2)+1);
% % clf
% % plot(hz,2*abs(cmwX(1:length(hz))))
% % 
% % %https://www.youtube.com/watch?v=4TTpwIZrUAo&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=3
% % % convolution in the time domain is equivalent to multiplication in the
% % % freq domain -> so in practice, we'll only ever do freq domain stuff cause
% % % faster
% % % wavelet convolution via frequency domain mutliplication:
% % % 1) Take fft of signal
% % % 2) mutliply fourier spectra point by point
% % % 3) take inverse fft
% % 
% % % so lets take some data and apply yhis
% % 
% % data=X{1}(:,1)';
% % %create wavelet like above like above
% % srate=600;
% % time=-2:1/srate:2;
% % frex=6.5; 
% % sine_wave=exp(1i*2*pi*frex.*time);
% % s=7/(2*pi*frex);
% % gaus_win=exp((-time.^2)./(2*s^2)); 
% % cmw=sine_wave .* gaus_win; 
% % 
% % %now define convolution parameters
% % nData=length(data);%length of sata
% % nKern=length(cmw);%length of wavelet
% % nConv=nData+nKern-1;%length of conolution outcome is always  data+wave-1
% % 
% % %now take fft of morlet wavelet
% % cmwX=fft(cmw,nConv);%by gigving it the n of the outcome, it does the zero padding itself
% % %and amplitude-normalise in the frequency domain (amplitude scale morlet wvelet by its peak)
% % cmwX=cmwX./max(cmwX);%mot completely necessary but makes sure that the outcome is gonnabe in the original units of the signal
% % 
% % %now take fft of data
% % dataX=fft(data,nConv);
% % 
% % %now do convolution
% % conv_res=cmwX.*dataX;  %because conv is mutliplication in freq fomain and we're now in fre domain(becuase we took fft of everything)
% % %also, usually we want to put this back into time domain after, so we can
% % %do this in one step conv_res=ifft(cmwX.*dataX), but here we're doing it
% % %step by step
% % 
% % %compute hz for plotting
% % hz=linspace(0,srate/2,floor(length(cmw)/2)+1);
% % figure
% % hold on
% % plot(hz,2*abs(dataX(1:length(hz))/length(data)))%plots freq of data
% % plot(hz,abs(cmwX(1:length(hz))))%plots wavelet in freq
% % plot(hz,2*abs(conv_res(1:length(hz))/length(data))); %plots producut of multiplying those two
% % 
% % %now lets get it back into the time domain
% % 
% % %keep in mind, the length of the result of the convolution is longer thn
% % %the signal
% % %so cut 1/2 of the length of the wavelet from beginning and end
% % half_wav=floor(length(cmw)/2)+1;
% % 
% % 
% % %take inverse fourier
% % conv_res_timedomain=ifft(conv_res);
% % conv_res_timedomain=conv_res_timedomain(half_wav-1:end-half_wav); %cut sides off (not data)
% % 
% % figure
% % plot(tVec,data,'k')
% % hold on
% % plot(tVec,real(conv_res_timedomain),'r')
% % 
% % 
% % %% ok that's all great, but now how do I get power and phase
% % %https://www.youtube.com/watch?v=A4M0cZSrHzY&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=4
% % %ok so we're using a complex morelt wvelet, so it has a real and an
% % %imaginary part (so thats just sine ans cosine, so a bit shifted). so when
% % %we convolute thatt, for each timepoint, we get a single complex number.
% % % complex numbers can be represented in like a coord system, where the y
% % % axis is now the imaginary axis and the x axis is now the real axis. then
% % % we can plot that number as a point in that space. Now if i plot the
% % % vecotr from the origin to that point, the length of the vector is
% % % amplitude abd the angle of that vector is phase angle.
% % % so at the end of an analysis, we get a single complex number for each
% % % time point and each frequency (obv only if we have a wavelet for each frequency)
% % %. The filtered EEG signal is the real part
% % % of those complex numbers
% % 
% % % so let's ise data and wavelet from above
% % 
% % as=conv_res_timedomain;
% % 
% % % lets plot the result from earlier in some now ways
% % figure(1), clf %this is the whole result of the convolution with imaginary and real parts over time
% % plot3(tVec,real(as),imag(as),'k')
% % xlabel('Time (ms)'), ylabel('real part'), zlabel('imaginary part')
% % rotate3d
% % 
% % figure(2), clf %this is the phase angle over time (and also ampliutde, both axis)
% % plot3(tVec,abs(as),angle(as),'k')
% % %2D: plot(tVec,angle(as),'k') % just phase)
% % xlabel('Time (ms)'), ylabel('Amplitude'), zlabel('Phase')
% % rotate3d
% % 
% % figure(3), clf
% % % plot the filtered signal (projection onto real axis)
% % subplot(311)
% % plot(tVec,real(as))
% % xlabel('Time (ms)'), ylabel('Amplitude (\muV)')
% % 
% % % angle(as) gives you phase per timepoints (for this particular freq)
% % % as is the timedomain result of the convultion
% % 
% % % Now if you have many trials:
% % % could do
% % % for freq=1:Freq
% % %   for trial=1:Trial
% % %       convolution
% % %   end
% % % end
% % 
% % % but we can actually get rid ot the trial loop!
% % % we can jsut concatenate all the trials (string them onto each other into
% % % one big trial), do our thing, and then split it up again, and then avg
% % % them. same result as doing each trial separately
% % %https://www.youtube.com/watch?v=wdrXzqgcYLM&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=5
% % %% this is the slow version with the trial loop
% % data=X{1};
% % % frequency parameters
% % min_freq =  2;
% % max_freq = 30;
% % num_frex = 40;
% % frex = linspace(min_freq,max_freq,num_frex);
% % 
% % 
% % % other wavelet parameters
% % range_cycles = [ 4 10 ];
% % 
% % s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);
% % wavtime = -2:1/srate:2;
% % half_wave = (length(wavtime)-1)/2;
% % 
% % tic; % start matlab timer
% % 
% % % FFT parameters
% % nWave = length(wavtime);
% % nData = size(data,1);
% % nConv = nWave + nData - 1;
% % 
% % % initialize output time-frequency data
% % tf = zeros(length(frex),size(data,1),size(data,2));
% % 
% % % loop over frequencies
% % for fi=1:length(frex)
% %     
% %     % create wavelet and get its FFT
% %     % the wavelet doesn't change on each trial...
% %     wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
% %     waveletX = fft(wavelet,nConv);
% %     waveletX = waveletX ./ max(waveletX);
% %     
% %     % now loop over trials...
% %     for triali=1:size(data,2)
% %         
% %         dataX = fft(data(:,triali)', nConv);
% %         
% %         % run convolution
% %         as = ifft(waveletX .* dataX);
% %         as = as(half_wave+1:end-half_wave);
% %     
% %         % put power data into big matrix
% %         tf(fi,:,triali) = abs(as).^2;
% %     end
% % end
% % 
% % tfTrialAve = mean(tf,3);
% % 
% % computationTime = toc; % end matlab timer for this cell
% % 
% % % plot results
% % figure(1), clf
% % contourf(tVec,frex,tfTrialAve,40,'linecolor','none')
% % 
% % %% this is the same but more efficient (no trial loop)
% % % we make one giant trial
% % data=X{1};
% % tic; % restart matlab timer
% % 
% % % FFT parameters
% % nWave = length(wavtime);
% % nData = size(data,1) * size(data,2); % This line is different from above!!
% % nConv = nWave + nData - 1;
% % 
% % % initialize output time-frequency data
% % tf = zeros(length(frex),size(data,1));
% % 
% % % now compute the FFT of all trials concatenated
% % alldata = reshape( data ,1,[]);
% % dataX   = fft( alldata ,nConv );
% % 
% % 
% % % loop over frequencies
% % for fi=1:length(frex)
% %     
% %     % create wavelet and get its FFT
% %     % the wavelet doesn't change on each trial...
% %     wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
% %     waveletX = fft(wavelet,nConv);
% %     waveletX = waveletX ./ max(waveletX);
% %     
% %     % now run convolution in one step
% %     as = ifft(waveletX .* dataX);
% %     as = as(half_wave+1:end-half_wave);
% %     
% %     % and reshape back to time X trials
% %     as = reshape( as, size(data,1), size(data,2) );
% %     
% %     % compute power and average over trials
% %     tf(fi,:) = mean( abs(as).^2 ,2);
% % end
% % 
% % computationTime(2) = toc; % end matlab timer
% % 
% % % plot results
% % figure(2), clf
% % contourf(tVec,frex,tf,40,'linecolor','none')
% %  
% % %hte onl diffference between those two comes from edge rtifacts, but doesnt
% % %matter
% % 
% % %% Use variable nr of wavelet cycles
% % %% this shows how they matter (byt fixed)
% % %ok now explain how we use wvaelets with different numberso f cycels for
% % %%the same analysis
% % 
% % %number of cycles: the larger the nr cycles, the worse the time precision
% % %and the better the freq precision (so maybe use fewer cycles for time
% % %precision of phase??) stay within 3 to 10 https://www.youtube.com/watch?v=fkQGnYUv-FI&list=PLn0OLiymPak2G__qvavn3T8k7R8ssKxVr&index=6
% % 
% % % mikexcohen@gmail.com
% % 
% % data=X{1};
% % % wavelet parameters
% % num_frex = 40;
% % min_freq =  2;
% % max_freq = 30;
% % 
% % 
% % 
% % % set a few different wavelet widths ("number of cycles" parameter)
% % num_cycles = [ 2 6 8 15 ]; %ths is a bit extreme. stay between 3 and 10
% % 
% % % other wavelet parameters
% % frex = linspace(min_freq,max_freq,num_frex);
% % time = -2:1/srate:2;
% % half_wave = (length(time)-1)/2;
% % 
% % % FFT parameters
% % nKern = length(time);
% % nData = size(data,1)*size(data,2);
% % nConv = nKern+nData-1;
% % 
% % % initialize output time-frequency data
% % tf = zeros(length(num_cycles),length(frex),size(data,1));
% % 
% % 
% % 
% % % FFT of data (doesn't change on frequency iteration)
% % dataX = fft(reshape(data,1,[]),nConv);
% % 
% % % loop over cycles
% % for cyclei=1:length(num_cycles) %so now  we're doing all of this for diffferent numbers of cycles
% %     
% %     for fi=1:length(frex)
% %         
% %         % create wavelet and get its FFT
% %         s = num_cycles(cyclei)/(2*pi*frex(fi));
% %         
% %         cmw  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
% %         cmwX = fft(cmw,nConv);
% %         cmwX = cmwX./max(cmwX);
% %         
% %         % run convolution, trim edges, and reshape to 2D (time X trials) 
% %         as = ifft(cmwX.*dataX,nConv);
% %         as = as(half_wave+1:end-half_wave);
% %         as = reshape(as,size(data,1),size(data,2));
% %         
% %         % put power data into big matrix
% %         tf(cyclei,fi,:) = mean(abs(as).^2,2);
% %     end
% %     
% %      
% % end
% % 
% % % plot results
% % figure(3), clf
% % for cyclei=1:length(num_cycles)
% %     subplot(2,2,cyclei)
% %     
% %     contourf(tVec,frex,squeeze(tf(cyclei,:,:)),40,'linecolor','none')
% % %     set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
% %     title([ 'Wavelet with ' num2str(num_cycles(cyclei)) ' cycles' ])
% %     xlabel('Time (ms)'), ylabel('Frequency (Hz)')
% % end
% % 
% % %% this is how to variable number of wavelet cycles
% % 
% % % set a few different wavelet widths (number of wavelet cycles)
% % range_cycles = [ 4 10 ];%min and max cycls
% % 
% % % other wavelet parameters
% % nCycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);%get 40  diff nr of cycles, logarithymically scaled
% % % so now, we have a different nr of cycles for each freq we defined
% % % earlier. so in the freq loop, it makes a new wavelet and grabs this nr of
% % % cycles out of here
% % 
% % % initialize output time-frequency data
% % tf = zeros(length(frex),size(data,1));
% % 
% % for fi=1:length(frex)
% %     
% %     % create wavelet and get its FFT
% %     s = nCycles(fi)/(2*pi*frex(fi));
% %     cmw = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
% %     
% %     cmwX = fft(cmw,nConv);
% %     
% %     % run convolution
% %     as = ifft(cmwX.*dataX,nConv);
% %     as = as(half_wave+1:end-half_wave);
% %     as = reshape(as,size(data,1),size(data,2));
% %     
% %     % put power data into big matrix
% %     tf(fi,:) = mean(abs(as).^2,2);
% % end
% % 
% % 
% % % plot results
% % figure(4), clf
% % subplot(2,2,1)
% % 
% % contourf(tVec,frex,tf,40,'linecolor','none')
% % % set(gca,'clim',[-3 3],'ydir','normal','xlim',[-300 1000])
% % title('Convolution with a range of cycles')
% % xlabel('Time (ms)'), ylabel('Frequency (Hz)')
% % 
% % %%
% % %% Now ITPC
% % % mikexcohen@gmail.com
% % 
% % %% Compute and plot TF-ITPC for one electrode
% % 
% % data=X{1};
% % 
% % % wavelet parameters
% % num_frex = 40;
% % min_freq =  2;
% % max_freq = 30;
% % 
% % 
% % % set range for variable number of wavelet cycles
% % range_cycles = [ 4 10 ];
% % 
% % % other wavelet parameters
% % frex = logspace(log10(min_freq),log10(max_freq),num_frex);
% % wavecycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
% % time = -2:1/srate:2;
% % half_wave_size = (length(time)-1)/2;
% % 
% % % FFT parameters
% % nWave = length(time);
% % nData = size(data,1)*size(data,2);
% % nConv = nWave+nData-1;
% % 
% % 
% % % FFT of data (doesn't change on frequency iteration)
% % dataX = fft( reshape(data,1,nData) ,nConv);
% % 
% % % initialize output time-frequency data
% % tf = zeros(num_frex,size(data,1));
% % 
% % % loop over frequencies
% % for fi=1:num_frex
% %     
% %     % create wavelet and get its FFT
% %     s = wavecycles(fi)/(2*pi*frex(fi));
% %     wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
% %     waveletX = fft(wavelet,nConv);
% %     
% %     % run convolution
% %     as = ifft(waveletX.*dataX,nConv);
% %     as = as(half_wave_size+1:end-half_wave_size);
% %     as = reshape(as,size(data,1),size(data,2));
% %     
% %     % compute ITPC
% %     tf(fi,:) = abs(mean(exp(1i*angle(as)),2)); %this transforms our phase angles into vectors in a polar plane - then takes the avergae of those vectors, and then the length of that average vector
% %     angles_all(fi,:,:)=angle(as);
% % end
% % 
% % % plot results
% % figure(1), clf
% % contourf(tVec,frex,tf,40,'linecolor','none')
% % % set(gca,'clim',[0 .6],'ydir','normal','xlim',[-300 1000])
% % title('ITPC')
% % 
% % 
% % angledata=squeeze(angles_all(1,1,:));
% % 
% % % compute ITPC and preferred phase angle
% % itpc      = abs(mean(exp(1i*angledata)));
% % prefAngle = angle(mean(exp(1i*angledata)));
% % 
% % 
% % % and plot...
% % figure(2), clf
% % 
% % % as linear histogram
% % subplot(3,3,4)
% % hist(angledata,20)
% % xlabel('Phase angle'), ylabel('Count')
% % set(gca,'xlim',[0 2*pi])
% % title([ 'Observed ITPC: ' num2str(itpc) ])
% % 
% % % and as polar distribution
% % subplot(1,2,2)
% % polar([zeros(1,200); angledata'],[zeros(1,200); ones(1,200)],'k')
% % hold on
% % h = polar([0 prefAngle],[0 itpc],'m');
% % set(h,'linew',3)
% % title([ 'Observed ITPC: ' num2str(itpc) ])
% % 
% % 
% % 
% % %% Now let's apply everything we just learned to trough_lock data
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % Fs=600;
% % % wavelet parameters
% % num_frex = 40;
% % min_freq =  2;
% % max_freq = 30;
% % 
% % data=X{1}';
% % data=rnd_50;
% % % data=trough_lock;
% % nr_trials=size(data,1);
% % nr_pnts=size(data,2);
% % 
% % channel2use = 'pz';
% % 
% % % set range for variable number of wavelet cycles
% % range_cycles = [ 4 10 ];
% % 
% % % other wavelet parameters
% % frex = logspace(log10(min_freq),log10(max_freq),num_frex);
% % wavecycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
% % time = -2:1/Fs:2; % this is he time for the wavelet, not the time of the data
% % half_wave_size = (length(time)-1)/2;
% % 
% % % FFT parameters
% % nWave = length(time);
% % nData = nr_pnts*nr_trials;
% % nConv = nWave+nData-1;
% % 
% % 
% % % FFT of data (doesn't change on frequency iteration)
% % % data here should be in timextrial format
% % dataX = fft( reshape(data',1,nData) ,nConv);
% % 
% % % initialize output time-frequency data
% % tf = zeros(num_frex,nr_pnts);
% % 
% % % loop over frequencies
% % for fi=1:num_frex
% %     
% %     % create wavelet and get its FFT
% %     s = wavecycles(fi)/(2*pi*frex(fi));
% %     wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));% exp(-time.^2./(2*s^2)); is just gaussian, so we'r emutliplying that sinusoid with a gaussian to get wavaelet
% %     waveletX = fft(wavelet,nConv);
% %     
% %     % run convolution
% %     as = ifft(waveletX.*dataX,nConv);
% %     as = as(half_wave_size+1:end-half_wave_size);
% %     as = reshape(as,nr_pnts,nr_trials);
% %     
% %     % compute ITPC
% %     tf(fi,:) = abs(mean(exp(1i*angle(as)),2));
% % %     tfr(fi,:) = mean(abs(as).^2,2);
% %     
% % end
% % 
% % 
% % % plot results
% % figure(1), clf
% % contourf(1:600,frex,tf,40,'linecolor','none')
% % 
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Fs=600;
% % % wavelet parameters
% % num_frex = 40;
% % min_freq =  2;
% % max_freq = 30;
% % 
% % data=X{1}';
% % data=rnd_50(1,:);
% % % data=trough_lock;
% % nr_trials=size(data,1);
% % nr_pnts=size(data,2);
% % 
% % channel2use = 'pz';
% % 
% % % set range for variable number of wavelet cycles
% % range_cycles = [ 4 10 ];
% % 
% % % other wavelet parameters
% % frex = logspace(log10(min_freq),log10(max_freq),num_frex);
% % wavecycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
% % time = -2:1/Fs:2;
% % half_wave_size = (length(time)-1)/2;
% % 
% % % FFT parameters
% % nWave = length(time);
% % nData = nr_pnts*nr_trials;
% % nConv = nWave+nData-1;
% % 
% % 
% % % FFT of data (doesn't change on frequency iteration)
% % % data here should be in timextrial format
% % dataX = fft( reshape(data',1,nData) ,nConv);
% % 
% % % initialize output time-frequency data
% % tf = zeros(num_frex,nr_pnts);
% % 
% % % loop over frequencies
% % for fi=1:num_frex
% %     
% %     % create wavelet and get its FFT
% %     s = wavecycles(fi)/(2*pi*frex(fi));
% %     wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
% %     waveletX = fft(wavelet,nConv);
% %     
% %     % run convolution
% %     as = ifft(waveletX.*dataX,nConv);
% %     as = as(half_wave_size+1:end-half_wave_size);
% %     as = reshape(as,nr_pnts,nr_trials);
% %     
% %     % compute ITPC
% %     tf(fi,:) = abs(mean(exp(1i*angle(as)),2));
% % %     tfr(fi,:) = mean(abs(as).^2,2);
% %     
% % end
% % 
% % 
% % % plot results
% % figure(1), clf
% % contourf(1:301,frex,tf,40,'linecolor','none')
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % contourf([tVec],frex,tfr,40,'linecolor','none')
% % set(gca,'clim',[0 .001],'ydir','normal')
% % title('ITPC')
% % 
% % 
% % 
% %     fVec=frex;
% %     S=X{1};
% %     S = S';
% %     TFR=nan(length(fVec),size(S,2));
% %         B = zeros(length(fVec),size(S,2)); 
% %         width=7;
% %         for i=1:size(S,1) % per trial         
% %             for j=1:length(fVec) %per freq
% %                 f=fVec(j);
% %                 s=detrend(S(i,:));
% %                 dt_s = 1/Fs;
% %                 sf = f/width;
% %                 st = 1/(2*pi*sf);
% %                 t=-3.5*st:dt_s:3.5*st;
% %                 A = 1/(st*sqrt(2*pi));
% %                 m = A*exp(-t.^2/(2*st^2)).*exp(1i*2*pi*f.*t);
% %                 y = conv(s,m);
% %                 y = 2*(dt_s*abs(y)).^2;
% %                 y = y(ceil(length(m)/2):length(y)-floor(length(m)/2));
% %                 B(j,:) = y + B(j,:);
% %             end
% %         end
% %         TFR = B/size(S,1);    
% % 
% % 
% %     imagesc([tVec],fVec,TFR)
% %     imagesc([tVec],fVec,tfr)
% %     
% %     
% %     
% %     
% %     
% %     
% %     
% % 
% %             
% % % specify parameters
% % circ_prop = .6; % proportion of the circle to fill
% % N = 100; % number of "trials"
% % 
% % % generate phase angle distribution
% % simdata = rand(1,N) * (2*pi) * circ_prop;
% % 
% % 
% % % compute ITPC and preferred phase angle
% % itpc      = abs(mean(exp(1i*simdata)));
% % prefAngle = angle(mean(exp(1i*simdata)));
% % 
% % 
% % % and plot...
% % figure(2), clf
% % 
% % % as linear histogram
% % subplot(3,3,4)
% % hist(simdata,20)
% % xlabel('Phase angle'), ylabel('Count')
% % set(gca,'xlim',[0 2*pi])
% % title([ 'Observed ITPC: ' num2str(itpc) ])
% % 
% % % and as polar distribution
% % subplot(1,2,2)
% % polar([zeros(1,N); simdata],[zeros(1,N); ones(1,N)],'k')
% % hold on
% % h = polar([0 prefAngle],[0 itpc],'m');
% % set(h,'linew',3)
% % title([ 'Observed ITPC: ' num2str(itpc) ])
% %             
% %             
% %             
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %% fieldtrip
% % addpath C:\Users\ckohl\Documents\fieldtrip-20190802\fieldtrip-20190802
% % ft_defaults
% % 
% % load('C:\Users\ckohl\Desktop\EEG')
% % for trial=1:length(EEG.trial)
% %     EEG.trial{trial}=EEG.trial{trial}(:,1:600);
% % %     EEG.time{trial}=EEG.time{trial}(:,1:600);
% %     EEG.time{trial}=EEG.time{trial}(1):dt./1000:(EEG.time{trial}(end)-1);
% % end
% % 
% % data=EEG;
% % data.fsample=Fs;
% % for trial=1:72
% %     data.time{trial}=tVec./1000;
% % %     data.trial{trial}=trough_lock(trial,:);
% % end
% % 
% % data=EEG;
% % % data.label=data.label(5);
% % % data.elec.label=data.label;
% % % data.elec.pnt=data.elec.pnt(5,:);
% % data.fsample=Fs;
% % tVec_s=[-1:dt/1000:0];
% % tVec_s(1)=[];
% % for trial=1:72
% %     data.time{trial}=tVec_s;
% % %     data.trial{trial}=trough_lock(trial,:);
% % end
% %     
% % 
% % cfg        = [];
% % cfg.method = 'wavelet';
% % cfg.toi    = data.time{trial};
% % cfg.foi= [5:1:30];
% % cfg.output = 'fourier';
% % cfg.output     = 'pow';
% % cfg.channel    = 'C3';
% % cfg.pad=2
% % freq       = ft_freqanalysis(cfg, data);
% % 
% % 
% % cfg = [];	                
% % cfg.method     = 'wavelet';                
% % cfg.width      = 7; %we want at least 4 apparently
% % cfg.output     = 'pow';	
% %     cfg.foi        = bandfreq(1):1:bandfreq(end);%10:1:30;%for short segments, use 3 as starting req for toi and foi we can be pretty generous. nothing to do with the time window and stuff	
% %     cfg.toi        = [-.75:.01:0];%EEG.time{1};
% % cfg.keeptrials = 'yes';
% % cfg.channel    = 'C3';
% % TFR = ft_freqanalysis(cfg, EEG);
% % cfg.keeptrials = 'no';
% % TFR_avg=ft_freqanalysis(cfg, EEG);
% % cfg=[];      
% %     cfg.parameter='powspctrm';
% %     cfg.colormap=jet;
% %     cfg.colorbar='yes';
% %     cfg.channel='C3';
% % %         cfg.baseline=[-1 0];
% % %         cfg.baselinetype ='relative';
% %     cfg.title = strcat('Avg');
% % ft_singleplotTFR(cfg,freq)
% % % make a new FieldTrip-style data structure containing the ITC
% % % copy the descriptive fields over from the frequency decomposition
% % 
% % itc           = [];
% % itc.label     = freq.label;
% % itc.freq      = freq.freq;
% % itc.time      = freq.time;
% % itc.dimord    = 'chan_freq_time';
% % 
% % 
% % F = freq.fourierspctrm;   % copy the Fourier spectrum
% % N = size(F,1);           % number of trials
% % 
% % % compute inter-trial phase coherence (itpc)
% % itc.itpc      = F./abs(F);         % divide by amplitude
% % itc.itpc      = sum(itc.itpc,1);   % sum angles
% % itc.itpc      = abs(itc.itpc)/N;   % take the absolute value and normalize
% % itc.itpc      = squeeze(itc.itpc); % remove the first singleton dimension
% % 
% % % compute inter-trial linear coherence (itlc)
% % itc.itlc      = sum(F) ./ (sqrt(N*sum(abs(F).^2)));
% % itc.itlc      = abs(itc.itlc);     % take the absolute value, i.e. ignore phase
% % itc.itlc      = squeeze(itc.itlc); % remove the first singleton dimension            
% %             
% %             
% %             
% %             
% %             
