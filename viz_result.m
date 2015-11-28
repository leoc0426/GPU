Sodtube_Rusanov_Results = load('Sodtube_Rusanov_Results.txt');

x = linspace(1, 200, 200);

figure(1);
set(gcf, 'Position', get(0,'ScreenSize'));

for index = 1:1:300    
    subplot(3,1,1);
    plot(x, Sodtube_Rusanov_Results((index-1)*200+1:(index)*200,1))
    title(['Temperature'])
    axis([1 200 -2 2])

    subplot(3,1,2);
    plot(x, Sodtube_Rusanov_Results((index-1)*200+1:(index)*200,2))
    title(['Velocity'])
    axis([1 200 -1 1])

    subplot(3,1,3);
    plot(x, Sodtube_Rusanov_Results((index-1)*200+1:(index)*200,3))
    title(['Pressure'])
    axis([1 200 0 1])
    
    getframe
end
