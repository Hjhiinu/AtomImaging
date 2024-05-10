clear
close all

% String definitions 
DirectoryString = 'path';
ImageFileBase = 'set1_a_';
LaserFileBase = 'set1_L_';
FileExtension = '.dat';

%Global Variables
ImageFile_number = 23;
%LaserImageNumber = 10;
LaserImageNumber = ImageFile_number;

% ImageFile name
ImageFileName = [DirectoryString ImageFileBase num2str(ImageFile_number) FileExtension];

% LaserFile name
LaserFileName = [DirectoryString LaserFileBase num2str(LaserImageNumber) FileExtension];

% Open Image and Laser Files
fid1 = fopen(ImageFileName,'r');
fid2 = fopen(LaserFileName,'r');

% Load Image into A, and Laser into B
A = fread(fid1,[640,480],'short');
B = fread(fid2,[640,480],'float');

% Close Image and Laser files
fclose(fid1);
fclose(fid2);

% Generate the ln(A/B): the atoms Image 
NonZeroArray = 0.001*ones(640,480);
%Atoms = log((B+NonZeroArray)./(A+NonZeroArray));
Atoms = 54.76E-12*log((B+NonZeroArray)./(A+NonZeroArray))*2*pi/(3*6.084E-13);

% Generate Region-of-Interest Atoms Image

%the entire width for left atom
X1 = 100; 
X2 = 500;   

%right atom
X3=340;
X4=400;

%the height should be the same
Y1 = 240;
Y2 = 440;

Y3 = 240;
Y4 = 440;
%Y2 = 435;

Atoms_ROI = Atoms(X1:X2,Y1:Y2);
%Atoms_ROI_atom2 = Atoms(X3:X4,Y3:Y4);

% Generate ROI integrated Atom Image cross-sections
% horizontal for left atom image
X1_h = X1+20;
Y1_h = 310;
X2_h = 305-20;
Y2_h = 370;

%horizontal for right atom image
X3_h = 305+20;
Y3_h = 310;
X4_h = 430-20;
Y4_h = 370;


% vertical for left atom
X1_v = 175;
Y1_v = Y1;
X2_v = 240;
Y2_v = Y2;

% vertical for right atom
X3_v = X3;
Y3_v = Y1;
X4_v = X4;
Y4_v = Y2;

% Blackout box bb_h
%X1_bb_h = 400;
Y1_bb_h = Y1;
X2_bb_h = 450;
Y2_bb_h = Y2;

% Blackout box bb_v
%X1_bb_v = 400;
Y1_bb_v = Y1;
X2_bb_v = 450;
Y2_bb_v = Y2;

% X Cross-Section (Integrated)
Atoms_ROI_h = Atoms(X1_h:X2_h,Y1_h:Y2_h);
Atoms_CrossSection_h = sum(Atoms_ROI_h');

Atoms_ROI_h_atom2 = Atoms(X3_h:X4_h,Y3_h:Y4_h);
Atoms_CrossSection_h_atom2 = sum(Atoms_ROI_h_atom2');

% X Cross-Section minus Blackout Box bb_h (Integrated)
Atoms_ROI_h_1 = Atoms(X1_h:X2_h,Y1_h:Y2_h);
Atoms_CrossSection_h_y_1 = sum(Atoms_ROI_h_1');
Atoms_CrossSection_h_x_1 = [X1_h:X2_h];
Atoms_ROI_h_2 = Atoms(X2_bb_h:500,Y1_h:Y2_h);
Atoms_CrossSection_h_y_2 = sum(Atoms_ROI_h_2');
Atoms_CrossSection_h_x_2 = [X2_bb_h:500];
Atoms_CrossSection_h_y = [Atoms_CrossSection_h_y_1 Atoms_CrossSection_h_y_2]';
Atoms_CrossSection_h_x = [Atoms_CrossSection_h_x_1 Atoms_CrossSection_h_x_2]';

Atoms_ROI_h_1_atom2 = Atoms(X3_h:X4_h,Y3_h:Y4_h);
Atoms_CrossSection_h_y_1_atom2 = sum(Atoms_ROI_h_1_atom2');
Atoms_CrossSection_h_x_1_atom2 = [X3_h:X4_h];
Atoms_ROI_h_2_atom2 = Atoms(X2_bb_h:500,Y3_h:Y4_h);
Atoms_CrossSection_h_y_2_atom2 = sum(Atoms_ROI_h_2_atom2');
Atoms_CrossSection_h_x_2_atom2 = [X2_bb_h:500];
Atoms_CrossSection_h_y_atom2 = [Atoms_CrossSection_h_y_1_atom2 Atoms_CrossSection_h_y_2_atom2]';
Atoms_CrossSection_h_x_atom2 = [Atoms_CrossSection_h_x_1_atom2 Atoms_CrossSection_h_x_2_atom2]';




% Y Cross-Section (Integrated)
Atoms_ROI_v = Atoms(X1_v:X2_v,Y1_v:Y2_v);
Atoms_CrossSection_v = sum(Atoms_ROI_v);

Atoms_ROI_v_atom2 = Atoms(X3_v:X4_v,Y3_v:Y4_v);
Atoms_CrossSection_v_atom2 = sum(Atoms_ROI_v_atom2);

% Y Cross-Section minus Blackout Box bb_v (Integrated)
Atoms_ROI_v_1 = Atoms(X1_v:X2_v,Y1_v:Y2_v);
Atoms_CrossSection_v_y_1 = sum(Atoms_ROI_v_1);
Atoms_ROI_v_2 = Atoms(X2_bb_v:500,Y1_v:Y2_v);
Atoms_CrossSection_v_y_2 = sum(Atoms_ROI_v_2);

Atoms_CrossSection_v_y = Atoms_CrossSection_v_y_1+Atoms_CrossSection_v_y_2;
Atoms_CrossSection_v_x = [Y1_v:Y2_v];


Atoms_ROI_v_1_atom2 = Atoms(X3_v:X4_v,Y3_v:Y4_v);
Atoms_CrossSection_v_y_1_atom2 = sum(Atoms_ROI_v_1_atom2);
Atoms_ROI_v_2_atom2 = Atoms(X2_bb_v:500,Y3_v:Y4_v);
Atoms_CrossSection_v_y_2_atom2 = sum(Atoms_ROI_v_2_atom2);
Atoms_CrossSection_v_y_atom2 = Atoms_CrossSection_v_y_1_atom2+Atoms_CrossSection_v_y_2;
Atoms_CrossSection_v_x_atom2 = [Y3_v:Y4_v];




% Data Analysis
% ------------------------------------------

% ATOM NUMBER
% **************************
% Atom Number Horizontal Cross-Section
Atoms_ROI_h_lhs = Atoms(X1_h:X1_v,Y1_h:Y2_h);
Atoms_ROI_h_rhs = Atoms(X2_v:X3_h,Y1_h:Y2_h);
Atoms_ROI_h_center = Atoms(X1_v:X2_v,Y1_h:Y2_h);
Atom_CrossSection_h_lhs = sum(Atoms_ROI_h_lhs');
Atom_CrossSection_h_rhs = sum(Atoms_ROI_h_rhs');
Atom_CrossSection_h_center = sum(Atoms_ROI_h_center');
background_h = [Atom_CrossSection_h_lhs Atom_CrossSection_h_rhs];
background_h_perpixel = mean(background_h);
AtomNumber_h = sum(Atom_CrossSection_h_center)-background_h_perpixel*numel(Atom_CrossSection_h_center);


Atoms_ROI_h_lhs_atom2 = Atoms(X3_h:X3_v,Y3_h:Y4_h);
Atoms_ROI_h_rhs_atom2 = Atoms(X3_v:X4_h,Y3_h:Y4_h);
Atoms_ROI_h_center_atom2 = Atoms(X3_v:X4_v,Y3_h:Y4_h);
Atom_CrossSection_h_lhs_atom2 = sum(Atoms_ROI_h_lhs_atom2');
Atom_CrossSection_h_rhs_atom2 = sum(Atoms_ROI_h_rhs_atom2');
Atom_CrossSection_h_center_atom2 = sum(Atoms_ROI_h_center_atom2');
background_h_atom2 = [Atom_CrossSection_h_lhs_atom2 Atom_CrossSection_h_rhs_atom2];
background_h_perpixel_atom2 = mean(background_h_atom2);
AtomNumber_h_atom2 = sum(Atom_CrossSection_h_center_atom2)-background_h_perpixel_atom2*numel(Atom_CrossSection_h_center_atom2);


% Atom Number Vertical Cross-Section
Atoms_ROI_v_low = Atoms(X1_v:X2_v,Y1_v:Y1_h);
Atoms_ROI_v_high = Atoms(X1_v:X2_v,Y2_h:Y2_v);
Atoms_ROI_v_center = Atoms(X1_v:X2_v,Y1_h:Y2_h);
Atom_CrossSection_v_low = sum(Atoms_ROI_v_low);
Atom_CrossSection_v_high = sum(Atoms_ROI_v_high);
Atom_CrossSection_v_center = sum(Atoms_ROI_v_center);
background_v = [Atom_CrossSection_v_low Atom_CrossSection_v_high];
background_v_perpixel = mean(background_v);
AtomNumber_v = sum(Atom_CrossSection_v_center)-background_v_perpixel*numel(Atom_CrossSection_v_center);



Atoms_ROI_v_low_atom2 = Atoms(X3_v:X4_v,Y3_v:Y3_h);
Atoms_ROI_v_high_atom2 = Atoms(X3_v:X4_v,Y4_h:Y4_v);
Atoms_ROI_v_center_atom2 = Atoms(X3_v:X4_v,Y3_h:Y4_h);
Atom_CrossSection_v_low_atom2 = sum(Atoms_ROI_v_low_atom2);
Atom_CrossSection_v_high_atom2 = sum(Atoms_ROI_v_high_atom2);
Atom_CrossSection_v_center_atom2 = sum(Atoms_ROI_v_center_atom2);
background_v_atom2 = [Atom_CrossSection_v_low_atom2 Atom_CrossSection_v_high_atom2];
%background_v = Atom_CrossSection_v_high;
background_v_perpixel_atom2 = mean(background_v_atom2);
AtomNumber_v_atom2 = sum(Atom_CrossSection_v_center_atom2)-background_v_perpixel_atom2*numel(Atom_CrossSection_v_center_atom2);



% Horizontal Width
% *****************************

% Make temporary matrix/vector for making a first guess at sigma
tmp = Atoms_CrossSection_h-background_h_perpixel*ones(size(Atoms_CrossSection_h));
tmp1 = tmp-0.1*max(tmp);
tmp2 = ((sign(tmp1)+1.0)/2.0).*tmp1;

amplitude_0 = max(Atoms_CrossSection_h);
%sigma_0 = cov(tmp2,1);
sigma_0 = 10.0;
%center_0 = sum(tmp2.*[X1_h:X2_h])/sum(tmp2); % center of mass guess
center_0 = 212;
background_0 = background_h_perpixel;




tmp_atom2 = Atoms_CrossSection_h_atom2-background_h_perpixel_atom2*ones(size(Atoms_CrossSection_h_atom2));
tmp1_atom2 = tmp_atom2-0.1*max(tmp_atom2);
tmp2_atom2 = ((sign(tmp1_atom2)+1.0)/2.0).*tmp1_atom2;

amplitude_0_atom2 = max(Atoms_CrossSection_h_atom2);
%sigma_0 = cov(tmp2,1);
sigma_0_atom2 = 10.0;
%center_0 = sum(tmp2.*[X1_h:X2_h])/sum(tmp2); % center of mass guess
center_0_atom2 = 365;
background_0_atom2 = background_h_perpixel_atom2;






Horizontal_gaussian_p0 = [amplitude_0;sigma_0;center_0;background_0];
Horizontal_data = [Atoms_CrossSection_h_x Atoms_CrossSection_h_y];
[Horizontal_gaussian_p1, horizontal_fit,deviation_h,Horizontal_jaco] = gaussian_fit(Horizontal_gaussian_p0,Horizontal_data);
Atom_Number_HorizontalFit = abs(sqrt(2*pi)*Horizontal_gaussian_p1(2)*Horizontal_gaussian_p1(1));
error_bar_h=sum(deviation_h)/sqrt((length(Horizontal_data)-1)*length(Horizontal_data));


Horizontal_gaussian_p0_atom2 = [amplitude_0_atom2;sigma_0_atom2;center_0_atom2;background_0_atom2];
Horizontal_data_atom2 = [Atoms_CrossSection_h_x_atom2 Atoms_CrossSection_h_y_atom2];
[Horizontal_gaussian_p1_atom2, horizontal_fit_atom2,deviation_h_atom2,Horizontal_atom2_jaco] = gaussian_fit(Horizontal_gaussian_p0_atom2,Horizontal_data_atom2);
Atom_Number_HorizontalFit_atom2 = abs(sqrt(2*pi)*Horizontal_gaussian_p1_atom2(2)*Horizontal_gaussian_p1_atom2(1));
error_bar_h_atom2=sum(deviation_h_atom2)/sqrt((length(Horizontal_data_atom2)-1)*length(Horizontal_data_atom2));





%Horizontal_gaussian_p1
%***************************************************************************************************************************

% Vertical Width
% *****************************

% Make temporary matrix/vector for making a first guess at sigma


tmp_atom2 = Atoms_CrossSection_v_atom2-background_v_perpixel_atom2*ones(size(Atoms_CrossSection_v_atom2));
tmp1_atom2 = tmp_atom2-0.1*max(tmp_atom2);
tmp2_atom2 = ((sign(tmp1_atom2)+1.0)/2.0).*tmp1_atom2;


tmp = Atoms_CrossSection_v-background_v_perpixel*ones(size(Atoms_CrossSection_v));
tmp1 = tmp-0.1*max(tmp);
tmp2 = ((sign(tmp1)+1.0)/2.0).*tmp1;

amplitude_0 = max(Atoms_CrossSection_v_y);
%sigma_0 = cov(tmp2,1);
sigma_0 = 10.0;
center_0 = 340;%sum(tmp2.*[Y1_v:Y2_v])/sum(tmp2); % center of mass guess
last_element = numel(Atoms_CrossSection_v_y);
background_0 = (Atoms_CrossSection_v_y(1)+ Atoms_CrossSection_v_y(2)+Atoms_CrossSection_v_y(3)+Atoms_CrossSection_v_y(4))/4.0;


amplitude_0_atom2 =max(Atoms_CrossSection_v_y_atom2);
%sigma_0 = cov(tmp2,1);
sigma_0_atom2 = 10.0;
center_0_atom2 =335; %sum(tmp2_atom2.*[Y3_v:Y4_v])/sum(tmp2_atom2); % center of mass guess
last_element_atom2 = numel(Atoms_CrossSection_v_y_atom2);
background_0_atom2 = (Atoms_CrossSection_v_y_atom2(1)+ Atoms_CrossSection_v_y_atom2(2)+Atoms_CrossSection_v_y_atom2(3)+Atoms_CrossSection_v_y_atom2(4))/4.0;






Vertical_gaussian_p0 = [amplitude_0;sigma_0;center_0;background_0];
Vertical_data = [Atoms_CrossSection_v_x' Atoms_CrossSection_v_y'];
[Vertical_gaussian_p1,vertical_fit] = gaussian_fit(Vertical_gaussian_p0,Vertical_data);

Vertical_gaussian_p0_atom2 = [amplitude_0_atom2;sigma_0_atom2;center_0_atom2;background_0_atom2];
Vertical_data_atom2 = [Atoms_CrossSection_v_x_atom2' Atoms_CrossSection_v_y_atom2'];
[Vertical_gaussian_p1_atom2,vertical_fit_atom2] = gaussian_fit(Vertical_gaussian_p0_atom2,Vertical_data_atom2);
%Vertical_gaussian_p1


% Atom Number from Vertical integrated cross-section gaussian fit
amplitude_0 = max(Atoms_CrossSection_v);
sigma_0 = 10.0;
center_0 = Vertical_gaussian_p0(3);
background_0 = (Atoms_CrossSection_v(1)+ Atoms_CrossSection_v(2)+Atoms_CrossSection_v(3)+Atoms_CrossSection_v(4))/4.0;

amplitude_0_atom2 = max(Atoms_CrossSection_v_atom2);
sigma_0_atom2 = 10.0;
center_0_atom2 = Vertical_gaussian_p0_atom2(3);
background_0_atom2 = (Atoms_CrossSection_v_atom2(1)+ Atoms_CrossSection_v_atom2(2)+Atoms_CrossSection_v_atom2(3)+Atoms_CrossSection_v_atom2(4))/4.0;

Vertical_data_1 = [[Y1_v:Y2_v]' Atoms_CrossSection_v'];
Vertical_gaussian_p0 = [amplitude_0;sigma_0;center_0;background_0];
[Vertical_gaussian_p2,vertical_fit,deviation_v,Vertical_jaco] = gaussian_fit(Vertical_gaussian_p0,Vertical_data_1);
Atom_Number_VerticalFit = abs(sqrt(2*pi)*Vertical_gaussian_p2(2)*Vertical_gaussian_p2(1));
%Atom_Number_VerticalFit = abs(sqrt(2*pi)*Vertical_gaussian_p0(2)*Vertical_gaussian_p0(1));


error_bar_v=sum(deviation_v)/sqrt((length(Vertical_data_1)-1)*length(Vertical_data_1));




Vertical_data_1_atom2 = [[Y3_v:Y4_v]' Atoms_CrossSection_v_atom2'];
Vertical_gaussian_p0_atom2 = [amplitude_0_atom2;sigma_0_atom2;center_0_atom2;background_0_atom2];
[Vertical_gaussian_p2_atom2,vertical_fit_atom2,deviation_v_atom2,Vertical_atom2_jaco] = gaussian_fit(Vertical_gaussian_p0_atom2,Vertical_data_1_atom2);
Atom_Number_VerticalFit_atom2 = abs(sqrt(2*pi)*Vertical_gaussian_p2_atom2(2)*Vertical_gaussian_p2_atom2(1));

error_bar_v_atom2=sum(deviation_v_atom2)/sqrt((length(Vertical_data_1_atom2)-1)*length(Vertical_data_1_atom2));


% End Data analysis
% -----------------------------------------------------------------------

filename='ExtractExcel.xlsx';
%python 
 writeMatricesToExcel(filename, Horizontal_gaussian_p0, Horizontal_data, Horizontal_gaussian_p0_atom2, Horizontal_data_atom2,Vertical_gaussian_p0,Vertical_data_1,Vertical_gaussian_p0_atom2,Vertical_data_1_atom2)

%Plot the data
%------------------------
scrsz = get(0,'ScreenSize');
figure('Position',[(scrsz(3)*0.1) (scrsz(4)*0.2) (scrsz(3)*0.7) (scrsz(4)*0.7)])

% plot image ROI
subplot(4,4,[1 2 5 6])
imagesc([X1;X2],[Y1,Y2],Atoms_ROI',[-40 200])

%imagesc([X3;X4],[Y3,Y4],Atoms_ROI_atom2',[-1 2])  %wrong


rectangle('Position',[X1_h,Y1_h,abs(X2_h-X1_h),abs(Y2_h-Y1_h)],'LineWidth',3,'EdgeColor','m')
rectangle('Position',[X1_v,Y1_v,abs(X2_v-X1_v),abs(Y2_v-Y1_v)],'LineWidth',3,'EdgeColor','m')

rectangle('Position',[X3_h,Y3_h,abs(X4_h-X3_h),abs(Y4_h-Y3_h)],'LineWidth',3,'EdgeColor','m')
rectangle('Position',[X3_v,Y3_v,abs(X4_v-X3_v),abs(Y4_v-Y3_v)],'LineWidth',3,'EdgeColor','m')


%blackbox below -only for reference
rectangle('Position',[X2_bb_h,Y1_bb_h,abs(X2_bb_h-500),abs(Y2_bb_h-Y1_bb_h)],'LineWidth',3,'LineStyle','--')
%rectangle('Position',[X1_bb_v,Y1_bb_v,abs(X2_bb_v-X1_bb_v),abs(Y2_bb_v-Y1_bb_v)],'LineWidth',3,'LineStyle',':','EdgeColor',[0.45 0.45 0.45])



% plot ROI integrated horizontal cross-sections

subplot(4,4,9)
plot([X1_h:X2_h],Atoms_CrossSection_h)
title('Left Atom (H), Atoms CrossSection h')  
ymin_h = min(Atoms_CrossSection_h)-0.1*abs(max(Atoms_CrossSection_h)-min(Atoms_CrossSection_h));
ymax_h = max(Atoms_CrossSection_h)+0.1*abs(max(Atoms_CrossSection_h)-min(Atoms_CrossSection_h));
axis([X1_h X2_h ymin_h ymax_h])

subplot(4,4,10)
plot([X3_h:X4_h],Atoms_CrossSection_h_atom2)
title('Right Atom (H), Atoms CrossSection h')  
ymin_h_atom2 = min(Atoms_CrossSection_h_atom2)-0.1*abs(max(Atoms_CrossSection_h_atom2)-min(Atoms_CrossSection_h_atom2));
ymax_h_atom2 = max(Atoms_CrossSection_h_atom2)+0.1*abs(max(Atoms_CrossSection_h_atom2)-min(Atoms_CrossSection_h_atom2));
axis([X3_h X4_h ymin_h_atom2 ymax_h_atom2])

subplot(4,4,13)
plot(Atoms_CrossSection_h_x,Atoms_CrossSection_h_y,'.',[X1_h:X2_h],gaussian(Horizontal_gaussian_p1,[X1_h:X2_h]),'-r')
title('Gaussian Fit(L) Atoms CrossSection (hx hy)')  
%plot(Atoms_CrossSection_h_x,Atoms_CrossSection_h_y,'.')
ymin_h = min(Atoms_CrossSection_h_y)-0.1*abs(max(Atoms_CrossSection_h_y)-min(Atoms_CrossSection_h_y));
ymax_h = max(Atoms_CrossSection_h_y)+0.1*abs(max(Atoms_CrossSection_h_y)-min(Atoms_CrossSection_h_y));
axis([X1_h X2_h ymin_h ymax_h])

subplot(4,4,14)
plot(Atoms_CrossSection_h_x_atom2,Atoms_CrossSection_h_y_atom2,'.',[X3_h:X4_h],gaussian(Horizontal_gaussian_p1_atom2,[X3_h:X4_h]),'-r')
title('Gaussian Fit(R) Atoms CrossSection (hx hy)')  
%plot(Atoms_CrossSection_h_x,Atoms_CrossSection_h_y,'.')
ymin_h_atom2 = min(Atoms_CrossSection_h_y_atom2)-0.1*abs(max(Atoms_CrossSection_h_y_atom2)-min(Atoms_CrossSection_h_y_atom2));
ymax_h_atom2 = max(Atoms_CrossSection_h_y_atom2)+0.1*abs(max(Atoms_CrossSection_h_y_atom2)-min(Atoms_CrossSection_h_y_atom2));
axis([X3_h X4_h ymin_h_atom2 ymax_h_atom2])

% plot ROI integrated vertical cross-sections
subplot(4,4,3)
plot(Atoms_CrossSection_v,[Y1_v:Y2_v],'.')
title('Left Atom (V), Atoms CrossSection h')  
%plot(Atoms_CrossSection_v,[Y1_v:Y2_v],'.',gaussian(Vertical_gaussian_p0,[Y1_v:Y2_v]),[Y1_v:Y2_v],'-r')
ymin_v = min(Atoms_CrossSection_v)-0.1*abs(max(Atoms_CrossSection_v)-min(Atoms_CrossSection_v));
ymax_v = max(Atoms_CrossSection_v)+0.1*abs(max(Atoms_CrossSection_v)-min(Atoms_CrossSection_v));
axis ij
axis([ymin_v ymax_v Y1_v Y2_v])

subplot(4,4,7)
plot(Atoms_CrossSection_v_atom2,[Y3_v:Y4_v],'.')
title('Right Atom (V), Atoms CrossSection h')  
%plot(Atoms_CrossSection_v,[Y1_v:Y2_v],'.',gaussian(Vertical_gaussian_p0,[Y1_v:Y2_v]),[Y1_v:Y2_v],'-r')
ymin_v_atom2 = min(Atoms_CrossSection_v_atom2)-0.1*abs(max(Atoms_CrossSection_v_atom2)-min(Atoms_CrossSection_v_atom2));
ymax_v_atom2 = max(Atoms_CrossSection_v_atom2)+0.1*abs(max(Atoms_CrossSection_v_atom2)-min(Atoms_CrossSection_v_atom2));
axis ij
axis([ymin_v_atom2 ymax_v_atom2 Y3_v Y4_v])

subplot(4,4,4)
plot(Atoms_CrossSection_v_y,Atoms_CrossSection_v_x,'.',gaussian(Vertical_gaussian_p1,[Y1_v:Y2_v]),[Y1_v:Y2_v],'-r')
title('Gaussian Fit(L) Atoms CrossSection (hy hx)')  
%plot(Atoms_CrossSection_v_y,Atoms_CrossSection_v_x,'.')
ymin_v = min(Atoms_CrossSection_v_y)-0.02*abs(max(Atoms_CrossSection_h_y)-min(Atoms_CrossSection_h_y));
ymax_v = max(Atoms_CrossSection_v_y)+0.02*abs(max(Atoms_CrossSection_h_y)-min(Atoms_CrossSection_h_y));
axis ij
axis([ymin_v ymax_v Y1_v Y2_v])

subplot(4,4,8)
plot(Atoms_CrossSection_v_y_atom2,Atoms_CrossSection_v_x_atom2,'.',gaussian(Vertical_gaussian_p1_atom2,[Y3_v:Y4_v]),[Y3_v:Y4_v],'-r')
title('Gaussian Fit(R) Atoms CrossSection (hy hx)')  
%plot(Atoms_CrossSection_v_y,Atoms_CrossSection_v_x,'.')
ymin_v_atom2 = min(Atoms_CrossSection_v_y_atom2)-0.02*abs(max(Atoms_CrossSection_h_y_atom2)-min(Atoms_CrossSection_h_y_atom2));
ymax_v_atom2 = max(Atoms_CrossSection_v_y_atom2)+0.02*abs(max(Atoms_CrossSection_h_y_atom2)-min(Atoms_CrossSection_h_y_atom2));
axis ij
axis([ymin_v_atom2 ymax_v_atom2 Y3_v Y4_v])

% Display Horizontal and Vertical Atom Numbers
textbox1_string = ['AtomNumber(L) h = ' num2str(AtomNumber_h)];
textbox1 = annotation('textbox',[0.5 0.45 0.25 0.03],'String',textbox1_string);

textbox2_string = ['AtomNumber(L) v = ' num2str(AtomNumber_v)];
textbox2 = annotation('textbox',[0.5 0.42 0.25 0.03],'String',textbox2_string);

textbox3_string = ['AtomNumber(L) h (fit) = ' num2str(Atom_Number_HorizontalFit+" ±"+error_bar_h)];
textbox3 = annotation('textbox',[0.5 0.39 0.25 0.03],'String',textbox3_string);

textbox4_string = ['AtomNumber(L) v (fit) = ' num2str(Atom_Number_VerticalFit+" ±"+error_bar_v)];
textbox4 = annotation('textbox',[0.5 0.36 0.25 0.03],'String',textbox4_string);


textbox5_string = ['AtomNumber(R) h = ' num2str(AtomNumber_h_atom2)];
textbox5 = annotation('textbox',[0.75 0.45 0.25 0.03],'String',textbox5_string);

textbox6_string = ['AtomNumber(R) v = ' num2str(AtomNumber_v_atom2)];
textbox6 = annotation('textbox',[0.75 0.42 0.25 0.03],'String',textbox6_string);

textbox7_string = ['AtomNumber(R) h (fit) = ' num2str(Atom_Number_HorizontalFit_atom2+" ±"+error_bar_h_atom2)];
textbox7 = annotation('textbox',[0.75 0.39 0.25 0.03],'String',textbox7_string);

textbox8_string = ['AtomNumber(R) v (fit) = ' num2str(Atom_Number_VerticalFit_atom2+" ±"+error_bar_v_atom2)];
textbox8 = annotation('textbox',[0.75 0.36 0.25 0.03],'String',textbox8_string);


Atom=(AtomNumber_h+AtomNumber_v)/2;
Atom_atom2=(AtomNumber_h_atom2+AtomNumber_v_atom2)/2;%average



totalAtom=(Atom_Number_HorizontalFit+Atom_Number_VerticalFit)/2;
totalAtom_atom2=(Atom_Number_HorizontalFit_atom2+Atom_Number_VerticalFit_atom2)/2;%average

errorbartotal_L=sqrt(error_bar_h^2+error_bar_v^2)/2;
errorbartotal_R=sqrt(error_bar_h_atom2^2+error_bar_v_atom2^2)/2;

textbox9_string = ['Fit avg Total num(L) = ' num2str(totalAtom+" ±"+errorbartotal_L)];
textbox9 = annotation('textbox',[0.5 0.33 0.25 0.03],'String',textbox9_string);

textbox10_string = ['Fit avg Total num(R) = ' num2str(totalAtom_atom2+" ±"+errorbartotal_R)];
textbox10 = annotation('textbox',[0.75 0.33 0.25 0.03],'String',textbox10_string);

textbox11_string = ['Theoretical error sqrt(N) (L)= ' num2str(sqrt(totalAtom))];
textbox11 = annotation('textbox',[0.5 0.3 0.25 0.03],'String',textbox11_string);

textbox12_string = ['Theoretical error sqrt(N) (R)= ' num2str(sqrt(totalAtom_atom2))];
textbox12 = annotation('textbox',[0.75 0.3 0.25 0.03],'String',textbox12_string);



total=totalAtom_atom2+totalAtom;
error_total=sqrt(errorbartotal_L^2+errorbartotal_R^2);

ratio_L=totalAtom/(total);
errorbar_L_ratio=ratio_L*sqrt((errorbartotal_L^2)/(totalAtom^2)+(error_total^2)/(total^2));

ratio_R=totalAtom_atom2/(total);
errorbar_R_ratio=ratio_R*sqrt((errorbartotal_R^2)/(totalAtom_atom2^2)+(error_total^2)/(total^2));

textbox13_string = ['Fit Fraction (L)/Total = ' num2str(ratio_L+" ±"+errorbar_L_ratio)];
textbox13 = annotation('textbox',[0.5 0.20 0.25 0.03],'String',textbox13_string);

textbox14_string = ['Fit Fraction (R)/Total = ' num2str(ratio_R+" ±"+errorbar_R_ratio)];
textbox14 = annotation('textbox',[0.75 0.20 0.25 0.03],'String',textbox14_string);

textbox15_string = ['Actual Fraction (L)/Total = ' num2str(Atom/(Atom+Atom_atom2))];
textbox15 = annotation('textbox',[0.5 0.17 0.25 0.03],'String',textbox15_string);

textbox16_string = ['Actual Fraction (R)/Total = ' num2str(Atom_atom2/(Atom+Atom_atom2))];
textbox16 = annotation('textbox',[0.75 0.17 0.25 0.03],'String',textbox16_string);

textbox17_string = ['Total Num= ' num2str(total)];
textbox17 = annotation('textbox',[0.75 0.14 0.25 0.03],'String',textbox17_string);

textbox18_string = ['Theoretical deviation at 0.5 fraction= ' num2str(1/sqrt(total))];
textbox18 = annotation('textbox',[0.5 0.14 0.25 0.03],'String',textbox18_string);

textbox19_string = ['Vertical Ratio_L' num2str(Atom_Number_VerticalFit/(Atom_Number_VerticalFit+Atom_Number_VerticalFit_atom2))];
textbox19 = annotation('textbox',[0.75 0.11 0.25 0.03],'String',textbox19_string);

textbox20_string = ['Horizontal Ratio_L' num2str(Atom_Number_HorizontalFit/(Atom_Number_HorizontalFit+Atom_Number_HorizontalFit_atom2))];
textbox20 = annotation('textbox',[0.5 0.11 0.25 0.03],'String',textbox20_string);

% New figure
figure(2)
imagesc([X1;X2],[Y1,Y2],Atoms_ROI',[-40 200])
set(gca,'dataAspectRatio',[1 1 1])
%axis equal

function writeMatricesToExcel(filename, varargin)
    % This function writes multiple matrices to an Excel file horizontally without overlapping.
    % filename: string representing the name and path of the Excel file
    % varargin: variable input arguments, each expected to be a matrix

    % Initialize the starting column
    startCol = 1;

    % Enhanced function to convert numeric column index to Excel column letters
    function excelCol = colNumToExcelCol(n)
        letters = '';
        while n > 0
            modVal = mod(n-1, 26);
            letters = [char(65 + modVal), letters]; % Concatenate in reverse order
            n = floor((n - modVal) / 26);
        end
        excelCol = letters;
    end

    % Loop through each matrix provided in varargin
    for idx = 1:length(varargin)
        % Current matrix
        currentMatrix = varargin{idx};

        % Write current matrix to Excel
        rangeStr = sprintf('%s1', colNumToExcelCol(startCol)); % Compute the range string
        writematrix(currentMatrix, filename, 'Sheet', 'Sheet1', 'Range', rangeStr);

        % Update startCol for the next matrix
        startCol = startCol + size(currentMatrix, 2) + 1; % Move start column beyond the current matrix
    end
end