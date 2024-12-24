% Evaluation of synthetic data with statistical similarity
% Evaluation methods : 1) Kolmogorov-Smirnov test, 2) Kullback–Leibler
% Divergence 3) Jensen-Shannon divergence
% Example data set : simulated synthetic data. v.s real data
clearvars;
%%% User defined variables %%%
RealData = 'Rdata.csv';
SyntheticData = 'Sdata.csv';
Nbootstrap = 20; % The number of bootstrapping
SPvalue = 0.9; % P value in significance of similarity test
SKScomplement = 0.9; % KS complement in significance of similarity test
plotflag1 = 1; % 1 for the plotting the results of KS-test, 0 for non-plotting
plotflag2 = 1; % 1 for the plotting the KL-divergence and JS-divergence
NumBins = 15; % The number of bins in data histogram for the KL-divergence calculation
ThreshKLDiv = 0.03; % The upper threshold of KL-divergence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rdata = readmatrix(RealData);
Sdata = readmatrix(SyntheticData);
for i = 1:Nbootstrap
    %for i = 1
    Rindex = randperm(length(Rdata), 200);
    d1 = Rdata(Rindex); d2 = Sdata(Rindex);
    %%% 2 sample KS-test
    [h,p,ks2stat] = kstest2(d1, d2);
    hAll(i) = h;
    pAll(i) = p;
    statAll(i) = ks2stat;
end

dall = [d1 d2]; mdall = mean(dall, 2);
% 1 sample t-test with statistics in KS-test
[hKSP, pKSP, ciKSP, statsKSP] = ttest(pAll,SPvalue,"Tail","right"); % in p value of KS-test
[hKSC, pKSC, ciKSC, statsKSC] = ttest(1-statAll,SKScomplement,"Tail","right"); % in statistics of KS-test
% plot distribution
if plotflag1
    figure1 = figure;
    axes1 = axes('Parent',figure1);
    hold(axes1,'on');
    h1 = histogram(d1,'DisplayName','Real data','Parent', axes1, 'NumBins', 15, 'BinLimits', [min(mdall) max(mdall)],'FaceAlpha',0.3, 'EdgeAlpha',0.1, 'FaceColor','r');
    h1.Normalization = 'probability';
    %h1Data = h1.Data; h1Count = h1.Values; h1Pr = h1Count./sum(h1Count); h1Bin = h1.BinEdges;
    hold on;
    h2 = histogram(d2, 'DisplayName','Synthetic data','Parent',axes1, 'NumBins', 15, 'BinLimits', [min(mdall) max(mdall)],'FaceAlpha',0.3, 'EdgeAlpha',0.1, 'FaceColor','b');
    h2.Normalization = 'probability';
    %h2Data = h2.Data; h2Count = h2.Values; h2Pr = h2Count./sum(h2Count); h2Bin = h2.BinEdges;
    xlabel('Volume'); ylabel('Probability');
    hold off;
    legend(axes1,'show');
    figure2 = figure;
    axes2 = axes('Parent',figure2);
    Err = [std(pAll) std(1-statAll)];
    data = [mean(pAll), mean(1-statAll)];
    xAxis = [1 2];
    bar(xAxis, data, 'FaceColor', [0.5 0.5 0.5]); hold on;
    er = errorbar(xAxis,data,Err./2,Err./2);
    set(axes2,'YLim', [0 1.1], 'XTick',[1 2],'XTickLabel',{'P value of KS-test','1-D of KS-test'});
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
end
%%% Kullback-Leibler Divergence
d1 = Rdata; d2 = Sdata;
%% Calculation of KL-divergence and
[h1Count,h1Edges] = histcounts(d1, NumBins, 'Normalization', 'probability');
h1Pr = h1Count./sum(h1Count);
h1Bin = h1Edges;
[h2Count,h2Edges] = histcounts(d2, NumBins, 'Normalization', 'probability');
h2Pr = h2Count./sum(h2Count);
h2Bin = h2Edges;
h1Pr = h1Pr + eps; h2Pr = h2Pr + eps;
h1Bin(1) = [];
KL = kldiv(h1Bin, h1Pr, h2Pr);
KLjs = kldiv(h1Bin, h1Pr, h2Pr, 'js');
if plotflag2
    figure3 = figure;
    axes3 = axes('Parent',figure3);
    hold(axes3,'on');
    h1 = histogram(d1,'DisplayName','Real data','Parent', axes3, 'NumBins', 15, 'BinLimits', [min(mdall) max(mdall)],'FaceAlpha',0.3, 'EdgeAlpha',0.1, 'FaceColor','r');
    h1.Normalization = 'probability';
    h1Data = h1.Data; h1Count = h1.Values; h1Pr = h1Count./sum(h1Count); h1Bin = h1.BinEdges;
    hold on;
    h2 = histogram(d2, 'DisplayName','Synthetic data','Parent',axes3, 'NumBins', 15, 'BinLimits', [min(mdall) max(mdall)],'FaceAlpha',0.3, 'EdgeAlpha',0.1, 'FaceColor','b');
    h2.Normalization = 'probability';
    h2Data = h2.Data; h2Count = h2.Values; h2Pr = h2Count./sum(h2Count); h2Bin = h2.BinEdges;
    xlabel('Volume'); ylabel('Probability');
    hold off;
    legend(axes3,'show');

end
function KL = kldiv(varValue,pVect1,pVect2,varargin)
if ~isequal(unique(varValue),sort(varValue)),
    warning('KLDIV:duplicates','X contains duplicate values. Treated as distinct values.')
end
if ~isequal(size(varValue),size(pVect1)) || ~isequal(size(varValue),size(pVect2)),
    error('All inputs must have same dimension.')
end
% Check probabilities sum to 1:
if (abs(sum(pVect1) - 1) > .00001) || (abs(sum(pVect2) - 1) > .00001),
    error('Probablities don''t sum to 1.')
end

if ~isempty(varargin),
    switch varargin{1},
        case 'js',
            logQvect = log2((pVect2+pVect1)/2);
            KL = .5 * (sum(pVect1.*(log2(pVect1)-logQvect)) + ...
                sum(pVect2.*(log2(pVect2)-logQvect)));

        case 'sym',
            KL1 = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
            KL2 = sum(pVect2 .* (log2(pVect2)-log2(pVect1)));
            KL = (KL1+KL2)/2;

        otherwise
            error(['Last argument' ' "' varargin{1} '" ' 'not recognized.'])
    end
else
    KL = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
end
end


