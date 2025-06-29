function lhs = lhs_u(u,DiE,BDiE,DiBt,perS,SS,SSt)
    lhs = BDiE*u;
    lhs(perS) = SS\(SSt\lhs(perS));
    lhs = u-DiE*u+DiBt*lhs;
end