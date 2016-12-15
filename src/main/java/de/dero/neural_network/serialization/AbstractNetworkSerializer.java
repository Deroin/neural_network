package de.dero.neural_network.serialization;

import com.google.gson.GsonBuilder;
import com.google.gson.typeadapters.RuntimeTypeAdapterFactory;
import de.dero.neural_network.activation.ActivationFunction;
import de.dero.neural_network.activation.HyperbolicTangent;

abstract class AbstractNetworkSerializer implements NetworkSerializer {

    private GsonBuilder builder;

    {
        builder = new GsonBuilder();
        registerType(builder, ActivationFunction.class, HyperbolicTangent.class);
    }

    GsonBuilder getGsonBuilder() {
        return builder;
    }

    @SafeVarargs
    private final <T> void registerType(GsonBuilder builder,
                                        Class<T> superClass,
                                        Class<? extends T>... childClasses) {
        RuntimeTypeAdapterFactory<T> factory = RuntimeTypeAdapterFactory.of(superClass);
        for (Class<? extends T> childClass : childClasses) {
            factory.registerSubtype(childClass);
        }
        builder.registerTypeAdapterFactory(factory);
    }
}
