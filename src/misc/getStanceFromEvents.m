function [stanceL,stanceR] = getStanceFromEvents(eventArray,eventLabels)

LHS=eventArray(:,strcmp('LHS',eventLabels));
RHS=eventArray(:,strcmp('RHS',eventLabels));
LTO=eventArray(:,strcmp('LTO',eventLabels));
RTO=eventArray(:,strcmp('RTO',eventLabels));


%% Left-leg:
stanceL=oneStance(LHS,LTO);

%% Right-leg
stanceR=oneStance(RHS,RTO);


end

function stance=oneStance(HS,TO)
aux=1*HS-1*TO;
stance=cumsum(aux);
if find(HS,1,'first')>find(TO,1,'first') %TO happens first, we start in stance
    stance=stance+1;
end

for i=1:length(stance)
    if stance(i)>1
        stance(i:end)=stance(i:end)-1;
        warning('Non consistent events detected. Fixing.')
    end
    if stance(i)<0
        stance(i:end)=stance(i:end)+1;
        warning('Non consistent events detected. Fixing.')
    end
end

end

