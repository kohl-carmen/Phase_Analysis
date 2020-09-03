%% ITPC analysis 
% This is the same as All_PhaseAngle.m, but using a Hilbert transform
% instead of a wavelet 

% This is currently just running on all trials, but you can run it on beta
% event trials (just like Beta_PhaseAngle.m) by uncommenting some chunks

% see Cohen.m for a breakdown of the ITPC part
% Using Shin's MEG detection data
% This script takes beta events closest to tap and finds the phase angle at
% 0 (=tap) to see if there's any phase clustering, comparing detected and
% non-detected conditions

clear
Partic=1:10;
ppt=1;

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
%     no_beta_trials=[];
%     for trial = 1:nr_trials
%         if any(trial==specEv_struct.Events.Events.trialind)
%             last_beta_i=max(find(specEv_struct.Events.Events.trialind==trial));
%             
%             max_t=specEv_struct.Events.Events.maximatiming(last_beta_i);
%             max_t_realtime=tVec(find(round(tVec_assumed,3)==round(max_t,3)));
%             
%             [trough,trough_i]=min(X{1}(max(1,find(tVec==max_t_realtime)-time_interval/2/dt):min(find(tVec==max_t_realtime)+time_interval/2/dt,length(tVec)),trial));  
%             temp=1:length(tVec);
%             temp=temp(max(1,find(tVec==max_t_realtime)-time_interval/2/dt):min(find(tVec==max_t_realtime)+time_interval/2/dt,length(tVec)));
%             trough_i=temp(trough_i);
%             
%             %time to look prev to trough
%             t_pre=50;
%             if trough_i>t_pre/dt
%                 trough_data_i=trough_i-t_pre/dt : length(tVec);
%                 
%                 trough_is(trial)=trough_i;
%                 trough_lock(trial,trough_data_i)=X{1}(trough_data_i,trial)';
%                 if plott
% %                     subplot(10,20,trial)
%                     subplot(5,1,[1:4])
%                     hold on
% %                     title(strcat('Trial',num2str(trial)))
%                     title(strcat('Partic',num2str(partic)))
%                     plot(tVec, trough_lock(trial,:),'Color',[.5 .5 .5])
%                     plot(tVec(trough_i), trough_lock(trial,trough_i),'ro')
%                 end
%             else
%                 no_beta_trials=[no_beta_trials,trial];
%             end
%         else
%             no_beta_trials=[no_beta_trials,trial];
%         end
%     end
%     if plott
%         subplot(5,1,5)
%         h=histogram(tVec(trough_is(~isnan(trough_is))));
%         h.FaceColor=[.5 .5 .5];
%         xlim([tVec(1) tVec(end)])
%     end
% 
%   
%    
%     %delete trials with no beta
%     trough_lock=trough_lock(1:size(X{1},2),:);
%     trough_is(nansum(trough_lock,2)==0)=[];    
%     beh(nansum(trough_lock,2)==0)=[];    
%     trough_lock(nansum(trough_lock,2)==0,:)=[];   
%     
%     og_trough_lock=trough_lock;
%     og_trough_is=trough_is;
%     % chop off ends to get rid of nan (this will cut off betas but who
%     % cares
%     % let's say I want at least 100ms
%     cutoff=100;
%     [temp,cutoff_i]=min(abs(tVec-(tVec(end)-cutoff)));
%     trough_lock=trough_lock(:,cutoff_i:end);
%     %delete though trials where not enough samples are in cutoff
%     delete=[];
%     for i=1:size(trough_lock,1)
%         if sum(trough_lock(i,1:5))==0
%             delete=[delete,i];
%         end
%     end
%     trough_lock(delete,:)=[];
%     trough_is(delete)=[];
%     beh(delete)=[];
%     og_trough_lock(delete,:)=[];
%     og_trough_is(delete,:)=[];
%     tVec_new=tVec(cutoff_i:end);
%       
%     
%     %% only keep if beta within X of tap
%     beta_distance=1000;
%     keepi=find(tVec(trough_is)>=tVec(end)-beta_distance);
%     trough_lock=trough_lock(keepi,:);
%     trough_is=trough_is(keepi);
%     og_trough_lock=og_trough_lock(keepi,:);
%     beh=beh(keepi);
%     subplot(5,1,[1:4])
%     hold on
%     title(strcat('Partic',num2str(partic), '-Remaining'))
%     for trial=1:size(trough_lock,1)
%         plot(tVec_new, trough_lock(trial,:),'Color','b')
%         plot(tVec(trough_is(trial)), og_trough_lock(trial,trough_is(trial)),'bo')
%     end
%     if ppt
%         print('-dpng','-r150',strcat('temp','.png'));
%         blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
%         Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
%         Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
%     end
    
    trough_lock=X{1}';    
    tVec_new=tVec;
        
    %split intp conds
    detect=trough_lock(beh==1,:);
    non=trough_lock(beh==0,:);
%     fprintf('\nTrials without events: %d\nTrials too short:  %d\n',length(no_beta_trials),length(delete))
    fprintf('\n\n-----\nRemaining:\nDetected:  %d\nNon-detected:  %d\n-----\n',size(detect,1),size(non,1))
    
    Conds={'detect','non'};
    for conds=1:2
        data=eval(Conds{conds});
        data=data';

    %% get phase
    
    % first, filter
    what_band=2; % 1=alpha, 2=beta
    alphaband= [8 13];
    betaband=[15 29];
    
    nyquist=Fs/2;
    if what_band==1
        filtbound=alphaband;
    else
        filtbound=betaband;
    end
    trans_width=.2;
    filt_order=round(2*(Fs/filtbound(1)));
    ffrequencies  = [ 0 (1-trans_width)*filtbound(1) filtbound (1+trans_width)*filtbound(2) nyquist ]/nyquist;
    idealresponse = [ 0 0 1 1 0 0 ];
    filterweights = firls(filt_order,ffrequencies,idealresponse);
    
    for trial=1:size(data,2)
        % apply filter to data
        filtered_data(:,trial) = filtfilt(filterweights,1,double(data(:,trial)));
        % take inverse FFT
        hilbertm = hilbert(filtered_data(:,trial));
        angles(:,trial)=angle(hilbertm);
    end
    
    figure('units','normalized','outerposition', [0 0 1 1]);
    count=0;
    for t=1:15:size(data,1)
        count=count+1;
        subplot(5,8,count)
        angledata=squeeze(angles(t,:));

        % compute ITPC and preferred phase angle
        itpc      = abs(mean(exp(1i*angledata)));
        prefAngle = angle(mean(exp(1i*angledata)));
            
            

        u=polar([zeros(1,size(angles,2)); angledata],[zeros(1,size(angles,2)); ones(1,size(angles,2))]);
        set(u,'Color',[.5 .5 .5]);
        hold on
        h = polar([0 prefAngle],[0 itpc],'r');
        set(h,'linew',4)
        if conds==1
            tit=sprintf('t=: %2.2f\nITPC-Detected: %2.2f',tVec(t), itpc);
        else
            tit=sprintf('t=: %2.2f\nITPC-NonDetected: %2.2f',tVec(t), itpc);
        end
        title(tit)
    end
       if ppt
        print('-dpng','-r150',strcat('temp','.png'));
        blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
        Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
        Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
       end
    
       
    figure('units','normalized','outerposition', [0 0 1 1]);
    count=0;
    for t=589:size(data,1)
        count=count+1;
        subplot(3,4,count)
        angledata=squeeze(angles(t,:));

        % compute ITPC and preferred phase angle
        itpc      = abs(mean(exp(1i*angledata)));
        prefAngle = angle(mean(exp(1i*angledata)));
            
            

        u=polar([zeros(1,size(angles,2)); angledata],[zeros(1,size(angles,2)); ones(1,size(angles,2))]);
        set(u,'Color',[.5 .5 .5]);
        hold on
        h = polar([0 prefAngle],[0 itpc],'r');
        set(h,'linew',4)
        if conds==1
            tit=sprintf('t=: %2.2f\nITPC-Detected: %2.2f',tVec(t), itpc);
        else
            tit=sprintf('t=: %2.2f\nITPC-NonDetected: %2.2f',tVec(t), itpc);
        end
        title(tit)
    end
       if ppt
        print('-dpng','-r150',strcat('temp','.png'));
        blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
        Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
        Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
    end
    close all      
    

    for t=1:size(data,1)
%         count=count+1;
%         subplot(6,10,count)
        angledata=squeeze(angles(t,:));

        % compute ITPC and preferred phase angle
        itpc      = abs(mean(exp(1i*angledata)));
        prefAngle = angle(mean(exp(1i*angledata)));
            
            

%         u=polar([zeros(1,size(angles,2)); angledata],[zeros(1,size(angles,2)); ones(1,size(angles,2))]);
%         set(u,'Color',[.5 .5 .5]);
%         hold on
%         h = polar([0 prefAngle],[0 itpc],'r');
%         set(h,'linew',4)
        
        KEEP_ITPC.(Conds{conds})(partic,t)=itpc;
        KEEP_ANGLE.(Conds{conds})(partic,t)=prefAngle;
    end
    

              
    
    
%     
%     
%     
%    % because my tria length varies per trial, i need a trial loop
%     %TIME x TRIAL
% %     data=trough_lock';
% % 
% %      data=rnd_50';
% %     data=trough_lock';
%     % wavelet parameters
%     num_frex = 15;
%     min_freq =  8;
%     max_freq = 29;
%     srate=Fs;
% 
% 
%     % set range for variable number of wavelet cycles
%     range_cycles = [ 4 10 ];
% 
%     % other wavelet parameters
%     frex = logspace(log10(min_freq),log10(max_freq),num_frex);
%     wavecycles = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex);
%     time = -2:1/srate:2;
%     half_wave_size = (length(time)-1)/2;
% 
%     % FFT parameters
%     nWave = length(time);
%     nData = size(data,1)*size(data,2);
%     nConv = nWave+nData-1;
% 
% 
%     % FFT of data (doesn't change on frequency iteration)
%     dataX = fft( reshape(data,1,nData) ,nConv);
% 
%     % initialize output time-frequency data
%     tf = zeros(num_frex,size(data,1));
%     angles_all= zeros(num_frex,size(data,1),size(data,2));
% 
%     % loop over frequencies
%     for fi=1:num_frex
% 
%         % create wavelet and get its FFT
%         s = wavecycles(fi)/(2*pi*frex(fi));
%         wavelet  = exp(2*1i*pi*frex(fi).*time) .* exp(-time.^2./(2*s^2));
%         waveletX = fft(wavelet,nConv);
% 
%         % run convolution
%         as = ifft(waveletX.*dataX,nConv);
%         as = as(half_wave_size+1:end-half_wave_size);
%         as = reshape(as,size(data,1),size(data,2));
% 
%         % compute ITPC
%         tf(fi,:) = abs(mean(exp(1i*angle(as)),2)); %this transforms our phase angles into vectors in a polar plane - then takes the avergae of those vectors, and then the length of that average vector
%         angles_all(fi,:,:)=angle(as);
%     end
% 
%     % plot results
%     figure(1), 
%     subplot(1,2,conds)
%     hold on
% %     contourf(1:size(tf,2),frex,tf,40,'linecolor','none')
%     try
%         contourf(tVec_new,frex,tf,40,'linecolor','none')
%     catch
%         contourf(1:size(tf,2),frex,tf,40,'linecolor','none')
%     end
%     set(gca,'clim',[0 .5])
% %     cbar
%     if conds==1
%         title(strcat('ITPC Detected  (',num2str(size(detect,1)),')'))
%     else
%         title(strcat('ITPC Nondetected (',num2str(size(non,1)),')'))
%     end
%   
%     if conds==2 & ppt
%         print('-dpng','-r150',strcat('temp','.png'));
%         blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
%         Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
%         Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
%     end
% 
%     %angles_all= freq x time x trial
%     figure('units','normalized','outerposition', [0 0 1 1]);
%     time_oi=590;
%     for freq_oi=1:length(frex)
%         angledata=squeeze(angles_all(freq_oi,time_oi,:));
% 
%         % compute ITPC and preferred phase angle
%         itpc      = abs(mean(exp(1i*angledata)));
%         prefAngle = angle(mean(exp(1i*angledata)));
% 
% 
% %         % and plot...
% %         figure(2), clf
% % 
% %         % as linear histogram
% %         subplot(3,3,4)
% %         hist(angledata,20)
% %         xlabel('Phase angle'), ylabel('Count')
% %         set(gca,'xlim',[0 2*pi])
% %         title([ 'Observed ITPC: ' num2str(itpc) ])
% 
%         % and as polar distribution
%         subplot(4,4,freq_oi)
%         u=polar([zeros(1,size(angles_all,3)); angledata'],[zeros(1,size(angles_all,3)); ones(1,size(angles_all,3))]);;
%         set(u,'Color',[.5 .5 .5]);
%         hold on
%         h = polar([0 prefAngle],[0 itpc],'r');
%         set(h,'linew',4)
%         if conds==1
%             tit=sprintf('Freq: %2.2f\nITPC-Detected: %2.2f',frex(freq_oi), itpc);
%         else
%             tit=sprintf('Freq: %2.2f\nITPC-NonDetected: %2.2f',frex(freq_oi), itpc);
%         end
%         title(tit)
%     end
%     
%     % ITPC per timepoint
%     %angles_all= freq x time x trial
%     for time_oi=1:size(angles_all,2)
%         for freq_oi=1:length(frex)
%             angledata=squeeze(angles_all(freq_oi,time_oi,:));
% 
%             % compute ITPC and preferred phase angle
%             itpc      = abs(mean(exp(1i*angledata)));
%             prefAngle = angle(mean(exp(1i*angledata)));
% 
%             KEEP_ITPC.(Conds{conds})(partic,freq_oi,time_oi)=itpc;
%             KEEP_ANGLE.(Conds{conds})(partic,freq_oi,time_oi)=prefAngle;
%         end
%     end
%     if ppt
%         print('-dpng','-r150',strcat('temp','.png'));
%         blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
%         Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
%         Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
%     end
%     TF_ALL.(Conds{conds})(:,:,partic)=tf;
%     ANGLES_ALL.(Conds{conds}).(strcat('P',num2str(partic)))=angles_all;
%     end
%     close all
% end
% 

% %% PLOT ITPC per partic,freq,time
% figure
% clf
% c=parula(length(Partic));
% for conds=1:length(Conds)  
%     subplot(1,2,conds)
%     for freq=1:length(frex)
%           for partic=1:length(Partic)
%         
%                 a=scatter3(tVec_new, ones(size(tVec_new)).*frex(freq), squeeze(KEEP_ITPC.(Conds{conds})(partic,freq,:)));
%                 a.Marker='.';
%                 a.MarkerEdgeColor=c(partic,:);
%                 
%                 hold on
%         end
%     end
%     title(Conds{conds})
%     xlabel('Time')
%     ylabel('Frequency')
%     zlabel('ITPC')
% end
% if ppt
%     print('-dpng','-r150',strcat('temp','.png'));
%     blankSlide = Presentation.SlideMaster.CustomLayouts.Item(7);
%     Slide1 = Presentation.Slides.AddSlide(1,blankSlide);
%     Image1 = Slide1.Shapes.AddPicture(strcat(cd,'/temp','.png'),'msoFalse','msoTrue',120,0,700,540);%10,20,700,500
% end
%             
%         
%         
% 
%  
%                 plot(tVec_new,squeeze(KEEP_ITPC.(Conds{conds})(partic,freq,:)),'-','Color',c(partic,:))
%             
%             all partics in one, each partic diff colour
%             
    
    
    end
end
    