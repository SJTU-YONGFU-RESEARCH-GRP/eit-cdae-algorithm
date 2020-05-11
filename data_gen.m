% run F:\eidors-v3.10\eidors\startup.m

% %parameters
bkgnd = 1;
% vim = zeros(256,99);
% vim_n = zeros(256,99);
img_out = ones(64,64,99);
%build commonmodel
imb=  mk_common_model('d2c',16);
img= mk_image(imb.fwd_model, bkgnd);
%%bkgnd 
vh= fwd_solve( img );
vhm = vh.meas';

% add elements
for i=301:400
    [x,x1] = find(img.fwd_model.elems == i);
    img.elem_data(x)=bkgnd/3;
    for j =1:18
        x2 = img.fwd_model.elems (x,:);
        [x3,x1] = find(img.fwd_model.elems == x2(j));
        x4((j-1)*6+1:6*j) = x3;
        img.elem_data(x3)=bkgnd/3;
    end
    vi= fwd_solve(img);
  % %     set for img
    hh=show_fem(img,[0 0 0]);
    calc_colours('cmap_type','greyscale');
    set(hh,'EdgeColor','[0 0 0]');
    set(hh,'LineStyle','none');
    axis square; axis off    
    opt.resolution   = 18;             
    print_convert('test1.png', opt);
    test1 = imread('test1.png');
    if i>301
        img_out(:,:,i-301) = test1(2:65,2:65,1);
        vimeas = vi.meas';
        % transfer for meas Mtrix
        vim(:,i-301) = [0,0,vimeas(1:13),0,0,0,0,vimeas(14:27),0,0,0,vimeas(28:41),...
            0,0,0,vimeas(42:55),0,0,0,vimeas(56:69),0,0,0,vimeas(70:83)...
            0,0,0,vimeas(84:97),0,0,0,vimeas(98:111),0,0,0,vimeas(112:125),0,0,0,vimeas(126:139)...
            0,0,0,vimeas(140:153),0,0,0,vimeas(154:167),0,0,0,vimeas(168:181),0,0,0,vimeas(182:195),0,0,0,0,vimeas(196:208),0,0];;   %208 value without noise
        
        %     AdRd -12dB SNR
        vi_n= vi;
        nvhl= std(vi.meas - vh.meas)*10^(-18/20);
        vi_n.meas = vi.meas + nvhl *randn(size(vi.meas));
        vi_nmeas = vi_n.meas';
        % transfer for meas Mtrix
        vim_n(:,i-301) = [0,0,vi_nmeas(1:13),0,0,0,0,vi_nmeas(14:27),0,0,0,vi_nmeas(28:41),...
            0,0,0,vi_nmeas(42:55),0,0,0,vi_nmeas(56:69),0,0,0,vi_nmeas(70:83)...
            0,0,0,vi_nmeas(84:97),0,0,0,vi_nmeas(98:111),0,0,0,vi_nmeas(112:125),0,0,0,vi_nmeas(126:139)...
            0,0,0,vi_nmeas(140:153),0,0,0,vi_nmeas(154:167),0,0,0,vi_nmeas(168:181),0,0,0,vi_nmeas(182:195),0,0,0,0,vi_nmeas(196:208),0,0];

        
    end
    
 %%reset value
    img.elem_data(x)=bkgnd; 
     img.elem_data(x4)=bkgnd; 

end


% transfer for meas Mtrix
vhh =[0,0,vhm(1:13),0,0,0,0,vhm(14:27),0,0,0,vhm(28:41),...
    0,0,0,vhm(42:55),0,0,0,vhm(56:69),0,0,0,vhm(70:83)...
    0,0,0,vhm(84:97),0,0,0,vhm(98:111),0,0,0,vhm(112:125),0,0,0,vhm(126:139)...
    0,0,0,vhm(140:153),0,0,0,vhm(154:167),0,0,0,vhm(168:181),0,0,0,vhm(182:195),0,0,0,0,vhm(196:208),0,0];
% 
fig = inv_solve(imb,vhh',vim_n);
rimg = calc_slices(fig);
rimg = calc_colours(rimg,fig); 
save('dataset','vim','vim_n','img_out','rimg','vhh')
