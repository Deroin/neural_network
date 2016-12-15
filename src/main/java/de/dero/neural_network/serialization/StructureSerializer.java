package de.dero.neural_network.serialization;

import com.google.gson.*;
import com.google.gson.internal.$Gson$Types;
import de.dero.neural_network.Connection;
import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.Weight;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Objects;

public class StructureSerializer extends AbstractNetworkSerializer {

    private final Gson gson;

    {
        GsonBuilder builder = getGsonBuilder();
        builder.registerTypeAdapter(
                $Gson$Types.newParameterizedTypeWithOwner(null, List.class, Weight.class),
                new WeightListTypeAdapter()
        );
        builder.registerTypeAdapter(Connection.class, new ConnectionTypeAdapter());
        builder.setExclusionStrategies(new ExclusionStrategy() {
            @Override
            public boolean shouldSkipField(FieldAttributes f) {
                return Objects.equals(f.getName(), "output");
            }

            @Override
            public boolean shouldSkipClass(Class<?> clazz) {
                return false;
            }
        });
        builder.setFieldNamingStrategy(f -> {
            switch (f.getName()) {
                case "inputConnections":
                    return "in";
            }
            return FieldNamingPolicy.IDENTITY.translateName(f);
        });
        gson = builder.create();
    }

    @Override
    public String fromNetwork(NeuralNetwork network) {
        return gson.toJson(network);
    }

    public NeuralNetwork toNetwork(String serializedNetwork) {
        return gson.fromJson(serializedNetwork, NeuralNetwork.class);
    }

    public NeuralNetwork toNetwork(InputStream serializedNetwork) throws IOException {
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(serializedNetwork))) {
            String line;
            StringBuilder builder = new StringBuilder();
            while ((line = bufferedReader.readLine()) != null) {
                builder.append(line);
            }

            return toNetwork(builder.toString());
        }
    }
}
