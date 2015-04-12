function [output] = millerTest(features,net,u,v,P,T)

PCs=size(P,1);
[s1,s2,s3,s4]=size(features);
output=zeros(2,s1,s3,s4);

for i=1:size(features,3)
    for j=1:size(features,4)
        testingData=features(:,:,i,j);
        N=size(testingData,1);
        reducedWindow=zeros(N,PCs);

        for k=T+1:N-T %TODO: more efficient implementation using conv (?)
            slidingWindow=testingData(k+[-T:T],:);
            reducedWindow(k,:)=P*((slidingWindow(:)-u')./v');
        end
        
        %Testing:
         output(:,:,i,j)= net(reducedWindow');
    end
end


end

