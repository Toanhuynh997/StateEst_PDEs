function ff1 = f1(x,y,t,mu, opt)
    % %% v1
    % ff1 = 0.5*sin(4*pi*(x))+0*t+0*x+0*y+0*mu;  
    
    if opt == 1
        ff1 = -0.5*cos(8*pi*(x))+0*t+0*x+0*y+0*mu;  % This 
        % ff1 = -0.1*cos(8*pi*(x))+0*t+0*x+0*y+0*mu;  % This 
    elseif opt == 2
        % ff1 = -0.5*cos(16*pi*(x/1))+0*t+0*x+0*y+0*mu;
        ff1 = -0.5*t*cos(16*pi*(x/1))+0*t+0*x+0*y+0*mu;
    end
    % %% v3
    % ff1 = 0.5*cos(12*pi*(sqrt(x+1)))+0*t+0*x+0*y+0*mu; % This ?
    % ff1 = -0.5*sin(16*pi*(x))+0*t+0*x+0*y+0*mu;  

    % %% v4
    % ff1 = -0.5*cos(16*pi*(x/1))+0*t+0*x+0*y+0*mu;  % This

    % ff1 = 0.5*sin(24*pi*(1/7-x/3))+0*t+0*x+0*y+0*mu; %v4

    % ff1 = -0.2*sin(24*pi*(2/9-x/3))+0*t+0*x+0*y+0*mu; %v5
end