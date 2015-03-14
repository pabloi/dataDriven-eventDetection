fc_3_2 = motionToFeetCenter(squeeze(motionArray(:, :, 3, 2)));
figure(1); hold off;
plotFCV(fc_3_2);
overlayEvents(1:10001, roundedEventArray(:, 1, 3, 2), -3, 3, 'k:');
overlayEvents(1:10001, roundedEventArray(:, 4, 3, 2), -3, 3, 'g:');
legend('ly', 'lz', 'lzv', 'LHS');