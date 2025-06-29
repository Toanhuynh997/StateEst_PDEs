function outImg = inpaint_task_MATLAB(idx, masked_img, img_mask)
    % inpaint_task_MATLAB - Inpaint one slice of masked_img.
    %
    %   idx         : The index of the slice in masked_img to inpaint
    %   masked_img  : A 3D array of shape (Nx, Ny, num_images)
    %   img_mask    : A 2D mask array (Nx, Ny), 1 where data is present,
    %                 0 where data is missing (or vice versa).

    fillRegion = (img_mask == 0);  % fill where mask==0 
    outImg = regionfill(masked_img(:,:,idx), fillRegion);
end