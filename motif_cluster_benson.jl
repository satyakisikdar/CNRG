# Find a motif-based cluster for any directed triangle motif.  
# Taken from https://gist.github.com/arbenson/a7bb06fb74977cddbb455e824519a55e
function MotifSpectralClust(A::SparseMatrixCSC{Int64,Int64}, motif::AbstractString)
    # Form motif adjacency matrix                                                                                                                                              
    B = min.(A, A')  # bidirectional links                                                                                                                                     
    U = A - B        # unidirectional links                                                                                                                                    
    if     motif == "M1"
        C = (U * U) .* U'
        W = C + C'
    elseif motif == "M2"
        C =  (B * U) .* U' + (U * B) .* U' + (U * U) .* B
        W = C + C'
    elseif motif == "M3"
        C = (B * B) .* U + (B * U) .* B + (U * B) .* B
        W = C + C'
    elseif motif == "M4"
        W = (B * B) .* B
    elseif motif == "M5"
        C = (U * U) .* U + (U * U') .* U + (U' * U) .* U
        W = C + C'
    elseif motif == "M6"
        W = (U * B) .* U + (B * U') .* U' + (U' * U) .* B
    elseif motif == "M7"
        W = (U' * B) .* U' + (B * U) .* U + (U * U') .* B
    else
        error("Motif must be one of M1, M2, M3, M4, M5, M6, or M7.")
    end

    # Get Fiedler eigenvector                                                                                                                                                  
    dinvsqrt = spdiagm(1.0 ./ sqrt.(vec(sum(W, 1))))
    NM = I - dinvsqrt * W * dinvsqrt
    lambdas, evecs = eigs(NM, nev=2, which=:SM)
    z = dinvsqrt * real(evecs[:, 2])

    # Sweep cut                                                                                                                                                                
    sigma = sortperm(z)
    C = W[sigma, sigma]
    Csums = sum(C, 1)'
    motifvolS = cumsum(Csums)
    motifvolSbar = sum(W) * ones(length(sigma)) - motifvolS
    conductances = cumsum(Csums - 2 * sum(triu(C), 1)') ./ min.(motifvolS, motifvolSbar)
    split = indmin(conductances)
    if split <= length(size(A, 1) / 2)
        return sigma[1:split]
    else
        return sigma[(split + 1):end]
    end
end
