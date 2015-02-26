function [feetCenter] = motionToFeetCenter(motion)

leftToe = motion(:, 1 + [0 2 4]*9);
rightToe = motion(:, 1 + [1 3 5]*9);
leftHeel = motion(:, 3 + [0 2 4]*9);
rightHeel = motion(:, 3 + [1 3 5]*9);

feetCenter = [ ((leftToe+leftHeel)/2) ((rightToe+rightHeel)/2) ];

end