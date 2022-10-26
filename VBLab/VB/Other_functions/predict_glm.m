function[Ylbl,YProb] = predict_glm(data,w,b)

    aux = data*w+b;
    YProb = 1./(1+exp(-aux));
    Ylbl = YProb;
    Ylbl(Ylbl>0.5) = 1;
    Ylbl(Ylbl<0.5) = 0;

end