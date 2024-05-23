struct GoMLException <: Exception
    var::String
end

Base.showerror(io::IO, e::GoMLException) = print(io, e.var)

