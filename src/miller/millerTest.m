function [output] = millerTest(features,net,u,v,P,T)

PCs=size(P,1);

for i=1:size(features,3)
    for j=1:size(features,4)
        testingData=features(:,:,i,j);
        N=size(testingData,1);
        reducedWindow=zeros(N,PCs);

        for k=T+1:N-T
            slidingWindow=testingData(k+[-T:T],:);
            reducedWindow(k,:)=P*((slidingWindow(:)-u')./v');
        end
        
        %Testing:
         output= net(reducedWindow');
    end
end


end

