function ff1 = f1_v2(x,y,t,mu)
    % %% v1
    % ff1 = 0.5*sin(4*pi*(x))+0*t+0*x+0*y+0*mu;  
    
    % %% v2
    % ff1 = -0.5*cos(8*pi*(x))+0*t+0*x+0*y+0*mu;  

    % %% v3
    % ff1 = 0.5*cos(12*pi*(x))+0*t+0*x+0*y+0*mu;  
    ff1 = -0.5*cos(24*pi*(x))+0*t+0*x+0*y+0*mu;  

    % %% v4
    % ff1 = -0.5*cos(24*pi*(1/7-x/3))+0*t+0*x+0*y+0*mu;  % 2 lines of
    % vorticesc

    % ff1 = -0.5*sin(24*pi*(2/9-x/2))+0*t+0*x+0*y+0*mu; %v5
end