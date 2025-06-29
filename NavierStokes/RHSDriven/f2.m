function ff2 = f2(x,y,t,mu, opt)

    if opt ==1
        ff2 = 0.5*sin(8*pi*(x))+0*t+0*x+0*y+0*mu;
        % ff2 = 0.1*sin(8*pi*(x))+0*t+0*x+0*y+0*mu;
    elseif opt == 2
        % ff2 =  0.5*sin(16*pi*(x/1))+0*t+0*x+0*y+0*mu;
        ff2 =  0.5*sin(16*pi*(x/1))+0*t+0*x+0*y+0*mu;
    end
    % %% v1
    % ff2 = -0.5*cos(4*pi*(x))+0*t+0*x+0*y+0*mu;

    % %% v2
    % ff2 = 0.5*sin(8*pi*(x))+0*t+0*x+0*y+0*mu;

    % %% v3
    % ff2 = -0.5*sin(12*pi*(sqrt(x+1)))+0*t+0*x+0*y+0*mu;
    % ff2 = -0.5*cos(16*pi*(x))+0*t+0*x+0*y+0*mu;

    % %% v4
    % ff2 =  0.5*sin(16*pi*(x/1))+0*t+0*x+0*y+0*mu;

    % ff2 =  -0.5*cos(24*pi*(1/7-x/3))+0*t+0*x+0*y+0*mu; %v4
    
    % ff2 =  -0.2*cos(24*pi*(2/9-x/3))+0*t+0*x+0*y+0*mu; %v5
end