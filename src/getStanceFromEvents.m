function [stanceL,stanceR] = getStanceFromEvents(LHS,RHS,LTO,RTO)

auxL=1*LHS-1*LTO;
stanceL=cumsum(auxL);
if find(LTO,1,'first')<find(LHS,1,'first') %First event is a TO, so we were in stance to begin with
    stanceL=stanceL+1;
end

auxR=1*RHS-1*RTO;
stanceR=cumsum(auxR);
if find(RTO,1,'first')<find(RHS,1,'first') %First event is a TO, so we were in stance to begin with
    stanceR=stanceR+1;
end



end

