function [predictedY] = interpretLSTMoutput(p)

[~,mlclass]=max(p,[],3);

stanceL= mod(mlclass,2);
stanceR= mlclass>1;

predictedY(:,:,1)=stanceL;
predictedY(:,:,2)=stanceR;
predictedY=permute(predictedY,[1,3,2]);

end

