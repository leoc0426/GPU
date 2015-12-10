clear; clc;
A = load('2DResults.txt');

X = reshape(A(1:100*100,1), 100, 100);
Y = reshape(A(1:100*100,2), 100, 100);

figure(1)
set(gcf, 'Position', get(0,'ScreenSize'));

for i=1:1:100
    D = reshape(A((100*100)*(i-1)+1:100*100*i,3), 100, 100);
    [FX,FY] = gradient(D,1,1);
    pcolor(X,Y,sqrt(FX.^2+FY.^2));
    shading flat;
    colormap(flipud(gray));
    caxis([0, 0.35]); 
    view(2);
    axis equal;
    title(['Change of density gradient of shock bubble']);
    getframe;
end
