function imageFeature = getGeodesicDistance(labelMask, nbrHood, geoImg, geoGamma)

[Wstar, starInfo] = getStarEdges(labelMask, nbrHood, geoImg, geoGamma);
imageFeature = starInfo.dFG;

