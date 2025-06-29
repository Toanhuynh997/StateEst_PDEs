function B = avg(A) % same as diff, but average
    if size(A,1)==1 
        B = (A(2:end)+A(1:end-1))/2;
    else            
        B = (A(2:end,:)+A(1:end-1,:))/2; 
    end
end