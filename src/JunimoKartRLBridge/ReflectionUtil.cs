using System.Collections;
using System.Collections.Concurrent;
using System.Reflection;
using Microsoft.Xna.Framework;

namespace JunimoKartRLBridge;

internal static class ReflectionUtil
{
    private static readonly ConcurrentDictionary<string, FieldInfo?> FieldCache = new();
    private static readonly ConcurrentDictionary<string, MethodInfo?> MethodCache = new();

    public static object? Field(object? target, string name)
    {
        if (target is null)
            return null;

        var type = target.GetType();
        var key = $"{type.AssemblyQualifiedName}|field|{name}";
        var field = FieldCache.GetOrAdd(key, _ => FindField(type, name));
        return field?.GetValue(target);
    }

    public static bool SetField(object? target, string name, object? value)
    {
        if (target is null)
            return false;

        var type = target.GetType();
        var key = $"{type.AssemblyQualifiedName}|field|{name}";
        var field = FieldCache.GetOrAdd(key, _ => FindField(type, name));
        if (field is null)
            return false;

        field.SetValue(target, value);
        return true;
    }

    public static T Field<T>(object? target, string name, T fallback = default!)
    {
        var value = Field(target, name);
        if (value is null)
            return fallback;

        try
        {
            if (value is T direct)
                return direct;

            if (typeof(T).IsEnum)
                return (T)Enum.ToObject(typeof(T), Convert.ToInt32(value));

            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return fallback;
        }
    }

    public static object? Property(object? target, string name)
    {
        if (target is null)
            return null;
        var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
        var property = target.GetType().GetProperty(name, flags);
        return property?.GetValue(target);
    }

    public static double DoubleProperty(object? target, string name, double fallback = 0)
    {
        var value = Property(target, name);
        if (value is null)
            return fallback;
        try
        {
            return Convert.ToDouble(value);
        }
        catch
        {
            return fallback;
        }
    }

    public static object? Invoke(object? target, string name, params object[] args)
    {
        if (target is null)
            return null;

        var type = target.GetType();
        var key = $"{type.AssemblyQualifiedName}|method|{name}|{args.Length}";
        var method = MethodCache.GetOrAdd(key, _ => FindMethod(type, name, args.Length));
        return method?.Invoke(target, args);
    }

    public static bool BoolMethod(object? target, string name, bool fallback = false)
    {
        var value = Invoke(target, name);
        return value is bool result ? result : fallback;
    }

    public static Vector2 VectorField(object? target, string name)
    {
        var value = Field(target, name);
        return value is Vector2 vector ? vector : Vector2.Zero;
    }

    public static int EnumId(object? value)
    {
        if (value is null)
            return -1;
        try
        {
            return Convert.ToInt32(value);
        }
        catch
        {
            return -1;
        }
    }

    public static IEnumerable<object> Enumerate(object? value)
    {
        if (value is null)
            yield break;

        if (value is IDictionary dictionary)
        {
            foreach (DictionaryEntry entry in dictionary)
            {
                if (entry.Value is IEnumerable nested)
                {
                    foreach (var item in nested)
                    {
                        if (item is not null)
                            yield return item;
                    }
                }
                else if (entry.Value is not null)
                {
                    yield return entry.Value;
                }
            }
            yield break;
        }

        if (value is IEnumerable enumerable and not string)
        {
            foreach (var item in enumerable)
            {
                if (item is not null)
                    yield return item;
            }
        }
    }

    public static bool InheritsTypeName(object? value, string typeName)
    {
        for (var type = value?.GetType(); type is not null; type = type.BaseType)
        {
            if (type.Name == typeName)
                return true;
        }
        return false;
    }

    public static BoundsSnapshot? Bounds(object? target)
    {
        var value = Invoke(target, "GetBounds");
        if (value is not Rectangle rectangle)
            return null;

        return new BoundsSnapshot
        {
            X = rectangle.X,
            Y = rectangle.Y,
            Width = rectangle.Width,
            Height = rectangle.Height
        };
    }

    public static VectorSnapshot Vector(Vector2 vector)
    {
        return new VectorSnapshot
        {
            X = vector.X,
            Y = vector.Y
        };
    }

    private static FieldInfo? FindField(Type type, string name)
    {
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
        for (var current = type; current is not null; current = current.BaseType)
        {
            var field = current.GetField(name, flags);
            if (field is not null)
                return field;
        }
        return null;
    }

    private static MethodInfo? FindMethod(Type type, string name, int argCount)
    {
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
        for (var current = type; current is not null; current = current.BaseType)
        {
            var method = current
                .GetMethods(flags)
                .FirstOrDefault(candidate => candidate.Name == name && candidate.GetParameters().Length == argCount);
            if (method is not null)
                return method;
        }
        return null;
    }
}
